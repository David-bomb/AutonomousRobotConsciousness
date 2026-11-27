"""Учебные продолжения и три независимые метки последствий одного физического хода."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import lru_cache
from heapq import heappop,heappush
import numpy as np
import torch
from maze.environment import Status
from maze.grid import locations,wave
from . import samples
from .adapter import MOVES
from .common import HERE,digest,group,read,write
from .phase_audit import restore

FIELDS=("core","grad","past","possible","legal","action","labels","win","source")


class Finish:
    """Точная достижимость с остатком ходов по графу первых контактов, только на train."""
    def __init__(self,env):
        self.stops={p:k for k in "FR" for p in locations(env.level.grid,k)}
        self.goal=env.level.goal
        self.stops[self.goal]="B"
        self.distance={p:d for p,d,_ in wave(env.level.grid,[self.goal])}
        self.legs={q:{p:d for p,d,_ in wave(env.level.grid,[q],stop_at=frozenset(self.stops))}
                   for q in self.stops}
        self.remaining=lru_cache(maxsize=32768)(self._remaining)

    def _remaining(self,start,fuel,health):
        if start==self.goal:return 0
        if min(fuel,health)<=0:return 10**9
        queue=[(0,start,fuel,health)];seen={}
        while queue:
            length,p,f,h=heappop(queue)
            if p==self.goal:return length
            labels=seen.setdefault(p,[])
            if any(x>=f and y>=h for x,y in labels):continue
            labels.append((f,h))
            for q,paths in self.legs.items():
                d=paths.get(p)
                if not d or d>min(f,h):continue
                nf=100 if self.stops[q]=="F" else f-d
                nh=100 if self.stops[q]=="R" else h-d
                if q==self.goal or min(nf,nh)>0: heappush(queue,(length+d,q,nf,nh))
        return 10**9


@torch.inference_mode()
def gather(row):
    name=row["name"];trace=read(HERE/"takeovers"/(name+".json"))
    extra=HERE/"mandatory-train"/(name+".json")
    sources=[trace["teacher"],read(extra) if extra.exists() else None]
    records=[]
    for source,trip in enumerate(sources):
        if not trip:continue
        env,previous=restore(row,trace)
        oracle=Finish(env)
        assert oracle.remaining(env.position,env.fuel,env.health)==trace["shortest_finish"]
        actions=trip["actions"];win=trip["outcome"]=="SUCCESS"
        rng=np.random.default_rng(int(row["geometry"][:14],16)+source)
        sample=set(range(min(20,len(actions))))
        pool=np.arange(20,len(actions))
        sample.update(rng.choice(pool,size=min(len(pool),128 if win else 64),replace=False).tolist())
        seen=set()
        for step,c in enumerate(actions):
            v=env.view;key=(v,previous);a=int(c)
            if step in sample and key not in seen:
                seen.add(key)
                core,grad,past,possible,legal=samples.NET.features(v,previous)
                labels=np.zeros((8,3),bool)
                for i in np.flatnonzero(legal):
                    sim=copy(env);sim._memory=copy(env._memory)
                    after=sim.step(MOVES[i])
                    assert sim.moved and env.view==v
                    viable=oracle.remaining(sim.position,sim.fuel,sim.health)<=sim.limit-sim.steps
                    labels[i]=(after.memory!=v.memory,oracle.distance[after.position]<oracle.distance[v.position],viable)
                records.append((core.astype("f2"),grad.astype("f2"),past.astype("f2"),possible,legal,a,labels,win,source))
            previous=v.fuel_distance,v.repair_distance
            env.step(MOVES[a]);assert env.moved
        assert env.status.name==trip["outcome"] and env.status!=Status.RUNNING
        oracle.remaining.cache_clear()
    dest=HERE/"samples/continuation"/(name+".npz");dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,**dict(zip(FIELDS,map(np.asarray,zip(*records)),strict=True)))
    return name,dict(sha256=digest(dest),examples=len(records),by_source=dict(Counter(r[-1] for r in records)))


def main():
    held=set(read(HERE/"terminal/route-search.json")["held_continuations"])
    book=read(HERE/"takeovers.json")
    batch=[r for r in group("extra")[:100] if r["name"] not in held and
           read(HERE/"takeovers"/(r["name"]+".json"))["reached"]]
    assert len(batch)==40 and len(held)==27
    with ProcessPoolExecutor(2,initializer=samples.initialize) as pool:
        result=dict(pool.map(gather,batch))
    write(HERE/"samples/continuation.json",dict(files=result,held_excluded=sorted(held),
          mandatory_sha256=digest(HERE/"mandatory.json"),takeover_sha256=digest(HERE/"takeovers.json"),
          labels=["new cell revealed","true road length to B decreases","physical finish reachable"],
          original_train_files={r["name"]:book["files"][r["name"]] for r in batch}))
    print("continuation examples",sum(v["examples"] for v in result.values()),flush=True)


if __name__=="__main__":main()
