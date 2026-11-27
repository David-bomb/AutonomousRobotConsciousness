"""Повтор учебных команд для привязки наблюдаемых границ к физической разметке."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import random
import numpy as np
from maze.environment import Environment,Status
from ..data import digest,level,rows,read,save
from ..policy import MOVES
from ..teacher import Teacher
from .alt_model import exits
from .bank import HERE
from .prepare import CACHE


def batch(job):
    keys,commands,original,output,prior=job
    if output.exists():
        return
    samples=[]
    for row,trace in zip(keys,commands):
        env=Environment(level(row))
        seen=set()
        expert=Teacher(env.level)
        rng=random.Random(int(row["geometry"][:12],16))
        while env.status==Status.RUNNING and env.steps<(len(trace) if trace is not None else 180):
            v=env.view
            best,viable=expert.labels(env)
            if trace is not None:
                a=int(trace[env.steps])
            else:
                from maze.grid import destination
                legal=np.array([destination(v.window,(1,1),m) is not None for m in MOVES])
                if best.any():
                    p=rng.random()
                    choices=legal if p<.035 else viable if p<.24 else best
                else:
                    choices=legal
                a=int(rng.choice(np.flatnonzero(choices)))
            if v not in seen:
                samples.append((exits(v,prior).astype(np.float16),best))
                seen.add(v)
            env.step(MOVES[a])
        expert.remaining.cache_clear()
    x,best=map(np.stack,zip(*samples))
    with np.load(original) as source:
        assert np.array_equal(best,source["best"])
    with output.with_suffix(".building").open("wb") as stream:
        np.savez_compressed(stream,exit=x)
    output.with_suffix(".building").replace(output)


def main():
    import torch
    import sys
    prior=torch.load(HERE/"runs/mission600/rl.pt",weights_only=True,map_location="cpu")["wall_prior"].numpy().tobytes()
    cfg=[]
    with ProcessPoolExecutor(2) as pool:
        for split,n in (("train",600),("clean",400),("probe",100)):
            src=read(HERE/f"prepare-{split}-{n}.json")
            train=rows(HERE/("probe.jsonl" if split=="probe" else "train.jsonl"))
            lookup={r["name"]:r for r in train}
            traces={r["map"]:r["actions"] for r in read(HERE/"clean-400.json")["episodes"]} if split=="clean" else {}
            jobs=[]
            for i,name in enumerate(src["parts"]):
                keys=src["names"][20*i:20*i+20]
                jobs.append(([lookup[k] for k in keys],[traces.get(k) for k in keys],CACHE/name,CACHE/("exits-"+name),prior))
            for i,_ in enumerate(pool.map(batch,jobs),1):
                if i%5==0:
                    print(split,20*i,flush=True)
            cfg.append(dict(split=split,n=n,parts={j[3].name:digest(j[3]) for j in jobs}))
    save(HERE/"exits-data.json",dict(sets=cfg,representation=digest(HERE/"alternatives.cpp"),dll=digest(HERE/"alternatives.dll")))


if __name__=="__main__":
    main()
