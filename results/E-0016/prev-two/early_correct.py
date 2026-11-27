"""Последовательные поправки на одной реальной учебной траектории."""
import os
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from . import early_data
from .adapter import MOVES,tensor_batch
from .common import HERE,TRAIN,DEV,PARENT,rows,level,digest,read,write
from .continuation_data import Finish
from .early_bank import FOLDER
from .early_policy import complete
from .mandatory import Previous
from ..data import generate


def register():
    path=FOLDER/"fit3.json"
    bank=generate(("prev-two-early-fit3",9176405))
    used={r["geometry"] for f in (TRAIN,DEV,PARENT/"data/train.jsonl") for r in rows(f)}
    for n in ("fit1","fit2","select","hold"):
        used.update(r["geometry"] for r in read(FOLDER/(n+".json")))
    assert len({r["geometry"] for r in bank})==100 and not (used&{r["geometry"] for r in bank})
    chosen=[r for i,cap in enumerate((8,8,25,25),1) for r in [x for x in bank if x["tier"]==f"H{i}"][:cap]]
    write(path,chosen)
    write(FOLDER/"correction-registration.json",dict(seed=9176405,bank=digest(path),plan=digest(HERE/"early-correction-plan.md"),
          parent=digest(HERE/"weights/early1200.pt"),source=digest(HERE/"early_correct.py")))
    return chosen


@torch.inference_mode()
def play(row):
    env=Environment(level(row));oracle=Finish(env);assert oracle.remaining(env.position,env.fuel,env.health)<=env.limit
    previous=(None,None);cache={};seen=set();records=[];interventions=[];actions=[];teachers=0
    while env.status==Status.RUNNING and not complete(env.view.memory):
        v=env.view;key=(v,previous)
        if key not in cache:
            core,grad,past,possible,legal=pack=early_data.NET.features(v,previous)
            logits=early_data.NET(*tensor_batch([pack]))[0].numpy()
            sign=np.zeros((8,2),np.int64);required=np.full(8,10**9)
            dist=np.array((v.fuel_distance,v.repair_distance));margins=np.array((v.fuel,v.health))-dist
            k=int(np.where(dist>0,margins,1000).argmin());urgent=1-k if v.health<v.fuel else k
            for j in np.flatnonzero(legal):
                child=copy(env);child._memory=copy(env._memory)
                after=child.step(MOVES[j]);assert child.moved and env.view==v
                d=np.array((after.fuel_distance,after.repair_distance))-dist+1
                sign[j]=d[::-1] if v.health<v.fuel else d
                required[j]=oracle.remaining(child.position,child.fuel,child.health)
                assert all(possible[j,g,sign[j,g]] for g in (0,1))
            cache[key]=(core.astype("f2"),grad.astype("f2"),past.astype("f2"),possible,legal,logits,sign,required,
                        urgent,int(margins[k]))
        core,grad,past,possible,legal,logits,sign,required,urgent,slack=cache[key]
        own=int(logits.argmax());viable=legal&(required<=env.limit-env.steps-1)
        assert viable.any(),(row["name"],env.steps)
        change=not viable[own];chosen=own;teach=-1
        if change:
            bot=Previous();bot.previous=previous if previous!=(None,None) else None
            suggestion=MOVES.index(bot.choose(v));teachers+=1
            if viable[suggestion]:chosen=suggestion;source="cold"
            else:
                # Учебный оценщик выбирает физически жизнеспособное продолжение с минимальной длиной до B.
                best=viable&(required==required[viable].min())
                chosen=int(np.where(best,logits,-1e9).argmax());source="exact"
            teach=chosen
            interventions.append(dict(step=env.steps,own=own,chosen=chosen,source=source,
                teacher=suggestion,viable=np.flatnonzero(viable).tolist(),required_after=required.tolist()))
        state=key,viable.tobytes()
        if state not in seen and (change or (legal.sum()>1 and len(records)<180)):
            seen.add(state)
            hard=change or (slack<=6 and sign[own,urgent]!=0 and ((sign[:,urgent]==0)&viable).any())
            records.append((core,grad,past,possible,legal,sign,viable,urgent,slack,teach,own,bool(hard),logits,
                            change,len(interventions)>int(change)))
        previous=v.fuel_distance,v.repair_distance
        env.step(MOVES[chosen]);assert env.moved;actions.append(str(chosen))
    base=FOLDER/"fit3";base.mkdir(exist_ok=True)
    out=base/(row["name"]+".npz")
    if records:
        np.savez_compressed(out,**dict(zip((*early_data.FIELDS,"correction","later"),map(np.asarray,zip(*records)),strict=True)))
    oracle.remaining.cache_clear()
    return row["name"],dict(tier=row["tier"],examples=len(records),sha256=digest(out) if records else None,
        actions="".join(actions),edits=interventions,windows=complete(env.view.memory),both=bool(env.refuels and env.repairs),
        parent="early1200",outcome=env.status.name,teacher_calls=teachers)


def main(workers=2):
    bank=register();result={}
    with ProcessPoolExecutor(workers,initializer=early_data.init,initargs=("early1200",)) as pool:
        for name,item in pool.map(play,bank):
            result[name]=item
            if len(result)%16==0:print("sequential corrections",len(result),flush=True)
    total=[len(r["edits"]) for r in result.values()]
    write(FOLDER/"fit3-samples.json",dict(parent="early1200",bank=digest(FOLDER/"fit3.json"),records=result,
          first=sum(n>0 for n in total),subsequent=sum(max(0,n-1) for n in total),total=sum(total),
          windows=sum(r["windows"] for r in result.values())))
    print("corrections",sum(n>0 for n in total),sum(total),flush=True)


if __name__=="__main__":main(*map(int,sys.argv[1:]))
