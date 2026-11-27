"""Фактические отклонения и разметка будущего только в учебных поездках."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from ..teacher import Teacher
from .adapter import MOVES,PrevNet,memory_fields,tensor_batch
from .common import BASE,BASE_SHA,HERE,SEQ,SEED,digest,group,level,read,write

NET=None


def initialize(checkpoint=None):
    global NET
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    state=torch.load(BASE,map_location="cpu",weights_only=True)
    NET=PrevNet("rich" if "base.wall_prior" in state else "critical" if "base.risk.0.weight" in state else "plain")
    NET.base.load_state_dict(state)
    if checkpoint: NET.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    NET.requires_grad_(False).eval()


@torch.inference_mode()
def prepare(job):
    kind,row,commands,checkpoint_sha=job
    folder=HERE/"samples"/kind
    folder.mkdir(exist_ok=True,parents=True)
    out=folder/(row["name"]+".npz")
    if out.exists():
        with np.load(out) as d: assert str(d["parent"])==checkpoint_sha
        return row["name"],digest(out)
    env=Environment(level(row)); teacher=Teacher(env.level)
    rng=np.random.default_rng(int(row["geometry"][:14],16))
    previous=(None,None)
    if commands:
        end=len(commands)
        record_idx=set(range(min(28,end)))
        others=np.arange(28,end)
        record_idx.update(rng.choice(others,size=min(128,len(others)),replace=False).tolist())
    else:
        end=160 if kind!="extra" else 220
        record_idx=set(range(end))
    collected=[]; seen=set()
    for t in range(end):
        if env.status!=Status.RUNNING: break
        v=env.view
        curr=(v.fuel_distance,v.repair_distance)
        key=v,previous
        keep=t in record_idx and key not in seen
        if keep or not commands:
            values=NET.features(v,previous)
            core,grad,past,possible,legal=values
        if commands:
            a=int(commands[t])
        elif rng.random() < (.18 if kind!="extra" else .12):
            a=int(rng.choice(np.flatnonzero(legal)))
        elif kind=="extra":
            a=int(NET(*tensor_batch([values])).argmax())
        else:
            a=int(np.where(legal,core[:,0],-1e6).argmax())
        if keep:
            best,viable=teacher.labels(env)
            phase=int(env.refuels>0)+int(env.repairs>0)
            alternative,abounds=memory_fields(v,(None,None),legal)
            back=-1
            if v.last_move and v.moved:
                back=MOVES.index(type(v.last_move)(tuple(-s for s in v.last_move.value)))
            seen.add(key)
        env.step(MOVES[a])
        assert env.moved
        if keep:
            labels=np.array([env.view.fuel_distance-curr[0]+1,env.view.repair_distance-curr[1]+1],np.int64)
            if v.health<v.fuel: labels=labels[::-1].copy()
            assert possible[a,0,labels[0]] and possible[a,1,labels[1]],(row["name"],t,labels,previous,curr)
            collected.append((core.astype(np.float16),grad.astype(np.float16),past.astype(np.float16),possible,legal,
                            alternative.astype(np.float16),abounds,a,labels,best,viable,phase,back,bool(commands)))
        previous=curr
    teacher.remaining.cache_clear()
    pending=out.with_suffix(".pending.npz")
    fields=("core","grad","past","possible","legal","missing","abounds","action","signs","best","viable","phase","back","demo")
    np.savez_compressed(pending,**{k:np.asarray(col) for k,col in zip(fields,zip(*collected),strict=True)},
                        parent=checkpoint_sha)
    pending.replace(out)
    return row["name"],digest(out)


def build(kinds,checkpoint=None):
    parent=digest(checkpoint) if checkpoint else BASE_SHA
    clean={r["map"]:r["actions"] for r in read(SEQ/"clean-400.json")["episodes"]}
    with ProcessPoolExecutor(2,initializer=initialize,initargs=(checkpoint,)) as pool:
        for kind in kinds:
            jobs=[(kind,row,clean[row["name"]] if kind=="clean" else None,parent) for row in group(kind)]
            result={}
            for name,sha in pool.map(prepare,jobs):
                result[name]=sha
                if len(result)%48==0: print(kind,len(result),"/",len(jobs),flush=True)
            write(HERE/"samples"/(kind+".json"),dict(parent=parent,files=result,sources={
                p:digest(HERE/p) for p in ("samples.py","adapter.py")}))


if __name__=="__main__":
    if sys.argv[1]=="extra":
        build(["extra"],HERE/"weights"/(sys.argv[2]+".pt"))
    else:
        build(["clean","deviate","probe"])
