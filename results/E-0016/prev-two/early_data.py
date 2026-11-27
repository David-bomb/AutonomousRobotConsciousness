"""Коррекции на ранних префиксах студента с исполнением соседних действий только на train."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from .adapter import MOVES,tensor_batch
from .common import HERE,level,read,write,digest
from .continuation_data import Finish
from .early_bank import FOLDER
from .early_policy import complete
from .fit import loadnet
from .mandatory import Previous

NET=None
FIELDS=("core","grad","past","possible","legal","sign","viable","urgent","slack","teacher","action","hard","prior")


def init(tag):
    global NET
    torch.set_num_threads(1)
    NET=loadnet(HERE/"weights"/(tag+".pt")).eval().requires_grad_(False)


@torch.inference_mode()
def play(job):
    row,batch,tag,idx=job
    path=FOLDER/batch/(row["name"]+".npz");path.parent.mkdir(exist_ok=True)
    env=Environment(level(row));oracle=Finish(env)
    rng=np.random.default_rng(int(row["geometry"][:14],16))
    previous=(None,None);seen={};records=[];actions=[];teacher_calls=0;teacher_errors=0
    while env.status==Status.RUNNING and not complete(env.view.memory) and len(actions)<360:
        v=env.view;key=v,previous
        if key in seen:
            a,legal=seen[key]
        else:
            core,grad,past,possible,legal=values=NET.features(v,previous)
            logits=NET(*tensor_batch([values]))[0].numpy()
            a=int(logits.argmax())
            seen[key]=(a,legal)
            distances=np.array((v.fuel_distance,v.repair_distance))
            margins=np.array((v.fuel,v.health))-distances
            k=int(np.where(distances>0,margins,1000).argmin())
            urgent=1-k if v.health<v.fuel else k
            if legal.sum()>1 and (len(records)<130 or (margins[k]<=6 and len(records)<210)):
                sign=np.zeros((8,2),np.int64);viable=np.zeros(8,bool)
                for j in np.flatnonzero(legal):
                    sim=copy(env);sim._memory=copy(env._memory)
                    after=sim.step(MOVES[j]);assert sim.moved
                    d=np.array((after.fuel_distance,after.repair_distance))-distances+1
                    sign[j]=d[::-1] if v.health<v.fuel else d
                    viable[j]=oracle.remaining(sim.position,sim.fuel,sim.health)<=sim.limit-sim.steps
                    assert all(possible[j,g,sign[j,g]] for g in (0,1))
                hard=(not viable[a] and viable.any()) or (margins[k]<=6 and sign[a,urgent]!=0 and
                                                         ((sign[:,urgent]==0)&viable&legal).any())
                teach=-1
                if teacher_calls<10 and (hard or (teacher_calls<5 and margins[k]<=10 and rng.random()<.10)):
                    ctrl=Previous();ctrl.previous=previous if previous!=(None,None) else None
                    taught=MOVES.index(ctrl.choose(v));teacher_calls+=1
                    if viable[taught]:teach=taught
                    else:teacher_errors+=1
                records.append((core.astype("f2"),grad.astype("f2"),past.astype("f2"),possible,legal,
                                sign,viable,urgent,int(margins[k]),teach,a,bool(hard),logits.astype("f4")))
        chosen=a
        if idx%3==1 and len(actions)>2 and rng.random()<.11:
            chosen=int(rng.choice(np.flatnonzero(legal)))
        previous=v.fuel_distance,v.repair_distance
        env.step(MOVES[chosen]);assert env.moved
        actions.append(str(chosen))
    oracle.remaining.cache_clear()
    if records:
        np.savez_compressed(path,**dict(zip(FIELDS,map(np.asarray,zip(*records)),strict=True)))
    return row["name"],dict(tier=row["tier"],parent=tag,examples=len(records),actions="".join(actions),
        windows=complete(env.view.memory),both=bool(env.refuels and env.repairs),outcome=env.status.name,
        hard=sum(bool(x[-2]) for x in records),teacher_calls=teacher_calls,teacher_infeasible=teacher_errors,
        sha256=digest(path) if records else None)


def collect(batch,tag,workers=2):
    lock=read(FOLDER/"registration.json");path=FOLDER/(batch+".json")
    assert digest(path)==lock["files"][path.name]
    if batch=="fit1":assert digest(HERE/"weights"/(tag+".pt"))==lock["wave1"]
    bank=read(path);records={}
    with ProcessPoolExecutor(int(workers),initializer=init,initargs=(tag,)) as pool:
        for name,record in pool.map(play,[(r,batch,tag,i) for i,r in enumerate(bank)]):
            records[name]=record
            if len(records)%16==0:print("early samples",batch,len(records),flush=True)
    write(FOLDER/(batch+"-samples.json"),dict(parent=tag,bank=digest(path),records=records))
    print("early data",batch,sum(v["examples"] for v in records.values()),flush=True)


if __name__=="__main__":collect(*sys.argv[1:])
