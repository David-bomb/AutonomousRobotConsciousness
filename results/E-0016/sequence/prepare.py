"""Физический сбор. Истинные будущие показания и учительские ходы остаются метками train."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import random
import time
import numpy as np
from maze.environment import Environment,Status
from maze.grid import destination
from .bank import HERE
from .model import encode
from ..policy import MOVES
from ..data import OUT,digest,level,read,rows,save
from ..priors import fit_prior
from ..teacher import Teacher

CACHE=HERE/"cache"


def build(job):
    output,batch,prior=job
    if output.is_file():
        with np.load(output) as data:
            return len(data["x"])
    samples=[]
    for row,commands in batch:
        seed=int(row["geometry"][:12],16)
        rng=random.Random(seed)
        env=Environment(level(row))
        expert=Teacher(env.level)
        seen=set()
        stats=Counter()
        while env.status==Status.RUNNING and env.steps<(len(commands) if commands is not None else 180):
            v=env.view
            best,viable=expert.labels(env)
            x,mask=encode(v,prior)
            gradient=np.full((8,2),1,np.int8)
            for a,m in enumerate(MOVES):
                if not mask[a]:
                    continue
                q=destination(env.level.grid,env.position,m)
                predicted_labels=[min(d[q] for d in expert.dist[:2])-v.fuel_distance,
                                  min(d[q] for d in expert.dist[2:4])-v.repair_distance]
                if v.health<v.fuel:
                    predicted_labels.reverse()
                assert all(abs(k)<=1 for k in predicted_labels)
                gradient[a]=np.array(predicted_labels)+1
            if commands is not None:
                a=int(commands[env.steps])
                assert mask[a]
            elif best.any():
                # Разрешённые отклонения учат и восстановлению, и прогнозу штатных измерений.
                prob=rng.random()
                choices=mask if prob<.035 else viable if prob<.24 else best
                a=int(rng.choice(np.flatnonzero(choices)))
            else:
                a=int(rng.choice(np.flatnonzero(mask)))
            slack=[v.fuel-v.fuel_distance,v.health-v.repair_distance]
            if v.health<v.fuel:
                slack.reverse()
            if v not in seen:
                samples.append((x.astype(np.float16),mask,best,viable,gradient,slack,
                                a if commands is not None and viable[a] else -1))
                seen.add(v)
            env.step(MOVES[a])
        expert.remaining.cache_clear()
    fields=("x","legal","best","viable","gradient","slack","partial")
    with output.with_suffix(".pending").open("wb") as temp:
        np.savez_compressed(temp,**{k:np.stack(v) for k,v in zip(fields,zip(*samples))})
    output.with_suffix(".pending").replace(output)
    return len(samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("split",choices=("train","probe","partial","clean"))
    p.add_argument("count",type=int)
    args=p.parse_args()
    assert args.count%20==0
    CACHE.mkdir(exist_ok=True)
    train=rows(OUT/"data/train.jsonl")
    prior=fit_prior(train)
    if args.split in ("partial","clean"):
        records=read(HERE/f"{args.split}-{args.count}.json")["episodes"]
        base=train if args.split=="partial" else rows(HERE/"train.jsonl")
        lookup={r["name"]:r for r in base}
        bank=[(lookup[r["map"]],r["actions"]) for r in records]
    else:
        bank=[(r,None) for r in rows(HERE/f"{args.split}.jsonl")[:args.count]]
    assert len(bank)==args.count and len({r["geometry"] for r,_ in bank})==args.count
    prefix=f"{args.split}{args.count}" if args.split in ("partial","clean") else args.split
    jobs=[(CACHE/f"{prefix}-{i:04}.npz",bank[i:i+20],prior.tobytes()) for i in range(0,args.count,20)]
    t=time.monotonic()
    with ProcessPoolExecutor(3) as pool:
        counts=[]
        for j,n in enumerate(pool.map(build,jobs)):
            counts.append(n)
            if (j+1)%5==0:
                print(prefix,(j+1)*20,sum(counts),round(time.monotonic()-t,1),flush=True)
    save(HERE/f"prepare-{args.split}-{args.count}.json",dict(count=args.count,records=sum(counts),
         names=[r["name"] for r,_ in bank],parts={p.name:dict(samples=n,sha256=digest(p)) for (p,_,_),n in zip(jobs,counts)},
         sources={n:digest(HERE/n) for n in ("prepare.py","model.py","geometry.cpp","sequence.dll")}))


if __name__=="__main__":
    main()
