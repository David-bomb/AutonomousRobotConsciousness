"""Геометрии train, реальные префиксы wave1 и проверка последствий физических действий."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from . import early_data
from .adapter import MOVES,tensor_batch
from .cautious_bank import OUT,bank
from .common import level,read,write,digest
from .continuation_data import Finish
from .early_policy import complete


def eligible(v,legal):
    distances=(v.fuel_distance,v.repair_distance)
    margins=(v.fuel-distances[0],v.health-distances[1])
    k=min(range(2),key=lambda i:margins[i] if distances[i]>0 else 1000)
    return k,margins[k],legal.sum()>1 and distances[k]>0 and 0<=margins[k]<=2


@torch.inference_mode()
def gather(row):
    out=OUT/"prefix"/(row["name"]+".json")
    if out.exists():return row["name"],digest(out)
    env=Environment(level(row));oracle=Finish(env);net=early_data.NET
    previous=(None,None);cache={};events=[];cmd=[];seen=set();edits=0
    while env.status==Status.RUNNING and not complete(env.view.memory):
        v=env.view;key=(v,previous)
        if key not in cache:
            pack=net.features(v,previous);legal=pack[-1]
            a=int(net(*tensor_batch([pack])).argmax())
            cache[key]=a,legal
        own,legal=cache[key];a=own
        k,slack,test=eligible(v,legal)
        if test and key not in seen:
            seen.add(key)
            signs=np.ones((8,2),int);required=np.full(8,10**8);target=np.array((v.fuel_distance,v.repair_distance))
            for i in np.flatnonzero(legal):
                child=copy(env);child._memory=copy(env._memory);after=child.step(MOVES[i]);assert child.moved
                signs[i]=np.array((after.fuel_distance,after.repair_distance))-target+1
                required[i]=oracle.remaining(child.position,child.fuel,child.health)
            viable=legal&(required<=env.limit-env.steps-1)
            if viable.any() and not viable[own]:
                best=viable&(required==required[viable].min())
                a=int(np.flatnonzero(best)[0]);edits+=1
            events.append(dict(step=env.steps,own=own,chosen=a,legal=legal.tolist(),urgent=k,slack=int(slack),
                               signs=signs.tolist(),viable=viable.tolist(),previous=list(previous)))
        previous=v.fuel_distance,v.repair_distance
        env.step(MOVES[a]);assert env.moved;cmd.append(str(a))
    oracle.remaining.cache_clear()
    write(out,dict(name=row["name"],geometry=row["geometry"],tier=row["tier"],actions="".join(cmd),
                   corrections=edits,events=events,windows=complete(env.view.memory),outcome=env.status.name))
    return row["name"],digest(out)


def main():
    maps=bank("fit")
    (OUT/"prefix").mkdir(exist_ok=True)
    traces={}
    with ProcessPoolExecutor(2,initializer=early_data.init,initargs=("wave1",)) as pool:
        for n,h in pool.map(gather,maps):
            traces[n]=h
            if len(traces)%80==0:print("conservative prefixes",len(traces),flush=True)
    write(OUT/"prefixes.json",dict(files=traces))


if __name__=="__main__":main()
