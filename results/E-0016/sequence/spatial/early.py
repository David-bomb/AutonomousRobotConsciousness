"""Первые физические развилки независимого train, с масками истинных кратчайших путей как учебной целью."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from maze.environment import Environment,Status
from maze.grid import destination,wave
from ...data import level,rows,save,digest
from ...policy import MOVES
from ...teacher import Teacher
from .model import HERE,image


def batch(job):
    bank,dest=job
    if dest.exists():return
    records=[]
    for r in bank:
        rng=np.random.default_rng(int(r["geometry"][:12],16))
        env=Environment(level(r))
        teach=Teacher(env.level)
        grid=env.level.grid
        side=40
        ends=np.full((3,side,side),10000,np.int16)
        for k,ds in enumerate((teach.dist[:2],teach.dist[2:4],teach.dist[4:])):
            for x,y in ds[0]:
                ends[k,y,x]=min(d[x,y] for d in ds)
        passable=ends[0]<10000
        # Два реальных старта на геометрию. Один префикс ближе к кратчайшему,
        # второй чаще содержит жизнеспособный обход и последующее восстановление.
        for noise in (.18,.6):
            env.reset()
            prev=None
            for step in range(65):
                v=env.view
                best,viable=teach.labels(env)
                assert viable.any()
                y=np.full((8,3),-1,np.int8)
                px,py=v.position
                actual=ends[:,py,px]
                neighbors=np.zeros(8,bool)
                for a,m in enumerate(MOVES):
                    q=destination(v.memory,v.position,m)
                    if q:
                        qx,qy=q
                        y[a]=1+ends[:,qy,qx]-actual
                        neighbors[a]=True
                critical=(neighbors&~viable).any()
                slack=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
                if neighbors.sum()>1 and (critical or slack<=8 or rng.random()<.2):
                    img,meta=image(v)
                    distance=np.full((side,side),10000,np.int16)
                    for (x,y0),d,_ in wave(grid,[v.position]):distance[y0,x]=d
                    target_path=(distance[None]+ends)==actual[:,None,None]
                    # Для обучения видимые и скрытые клетки области имеют
                    # одинаковую геометрическую разметку. Карта оценки остаётся скрытой.
                    dense=(passable.astype(np.uint8)*8)|(target_path[0].astype(np.uint8))|(
                        target_path[1].astype(np.uint8)*2)|(target_path[2].astype(np.uint8)*4)
                    records.append((img,meta,y,viable,best,dense,(v.fuel,v.health),
                                    MOVES.index(v.last_move) if v.last_move is not None else 8))
                choices=np.flatnonzero(viable if rng.random()<noise else best)
                if len(choices)>1 and prev is not None and rng.random()<.8:
                    forward=[a for a in choices if destination(v.memory,v.position,MOVES[a])!=prev]
                    if forward:choices=forward
                prev=v.position
                env.step(MOVES[int(rng.choice(choices))])
                if env.status!=Status.RUNNING or (env.refuels and env.repairs):break
        teach.remaining.cache_clear()
    with dest.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(v) for k,v in zip(
            ("maps","meta","y","viable","best","dense","stocks","last"),zip(*records))})
    dest.with_suffix(".tmp").replace(dest)


def main():
    source=HERE/"extra.jsonl"
    bank=rows(source)
    folder=HERE.parent/"cache/spatial"
    groups=[]
    for name,subset in (("early",bank[:3000]),("early-held",bank[3000:])):
        jobs=[(subset[i:i+20],folder/f"{name}-{i:04}.npz") for i in range(0,len(subset),20)]
        with ProcessPoolExecutor(3) as pool:
            for j,_ in enumerate(pool.map(batch,jobs),1):
                if j%20==0:print(name,j*20,flush=True)
        states=hard=0
        for _,p in jobs:
            with np.load(p) as f:
                states+=len(f["meta"]);hard+=int(((f["y"][:,:,0]>=0)&~f["viable"]).any(1).sum())
        groups.append(dict(split=name,maps=len(subset),states=states,critical=hard,parts={p.name:digest(p) for _,p in jobs}))
    save(HERE/"early.json",dict(train_sha256=digest(source),sets=groups,source_sha256=digest(HERE/"early.py"),
          prefixes=2,max_steps=65,viable_deviations=[.18,.6]))


if __name__=="__main__":
    main()
