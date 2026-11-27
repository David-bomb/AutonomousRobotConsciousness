"""Собственные учебные развилки с исправлением необратимых команд только при сборе."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import digest,level,rows,save
from ...policy import MOVES
from ...teacher import Teacher
from ..alt_model import ExitNet,exits
from ..model import encode
from .features import HERE,features


def init():
    global net,prior,tables
    torch.set_num_threads(1)
    w=HERE.parent/"runs/exit600/s900.pt"
    net=ExitNet().eval()
    net.load_state_dict(torch.load(w,weights_only=True,map_location="cpu"))
    prior=net.wall_prior.numpy().tobytes()
    tables=np.load(HERE/"prior.npy").tobytes()


@torch.inference_mode()
def collect(job):
    output,batch=job
    if output.exists():
        return
    records=[]
    for r in batch:
        rng=np.random.default_rng(int(r["geometry"][:12],16))
        env=Environment(level(r))
        teacher=Teacher(env.level)
        visited=set()
        unchanged=0
        corrections=0
        while env.status==Status.RUNNING and env.steps<280:
            v=env.view
            best,viable=teacher.labels(env)
            assert viable.any()
            x,mask=encode(v,prior)
            e=exits(v,prior)
            score=net.full(torch.from_numpy(x[None]),torch.from_numpy(e[None]),torch.from_numpy(mask[None]))[0][0].numpy()
            default=int(score.argmax())
            tight=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
            critical=(mask&~viable).any()
            a=default
            if not viable[a] or unchanged>=14:
                a=int(rng.choice(np.flatnonzero(best)))
                corrections+=1
            elif rng.random()<.03:
                a=int(rng.choice(np.flatnonzero(viable)))
            if mask.sum()>1 and v not in visited and (critical or tight<=7 or rng.random()<.3):
                y=np.ones((8,2),np.int8)
                for j in np.flatnonzero(mask):
                    q=destination(v.memory,v.position,MOVES[j])
                    y[j]=[1+min(ds[q] for ds in teacher.dist[:2])-v.fuel_distance,
                          1+min(ds[q] for ds in teacher.dist[2:4])-v.repair_distance]
                slack=np.array([v.fuel-v.fuel_distance,v.health-v.repair_distance],np.int16)
                if v.health<v.fuel:
                    y=y[:,::-1]
                    slack=slack[::-1]
                records.append((x.astype(np.float16),e.astype(np.float16),features(v,tables).astype(np.float16),mask,
                                best,viable,y,slack,default,int(env.refuels>0)+int(env.repairs>0)))
                visited.add(v)
            env.step(MOVES[a])
            unchanged=unchanged+1 if env.view.memory==v.memory else 0
        teacher.remaining.cache_clear()
    with output.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(column) for k,column in zip(
            ("x","exit","walk","ok","best","viable","sign","slack","default","phase"),zip(*records))})
    output.with_suffix(".pending").replace(output)


def main(n):
    assert n%100==0 and 0<n<=800
    bank=rows(HERE.parent/"train.jsonl")
    selected=(bank[400:900]+bank[1100:1400])[:n]
    path=HERE.parent/"cache/belief"
    path.mkdir(exist_ok=True)
    jobs=[(path/f"train-{i:04}.npz",selected[i:i+10]) for i in range(0,n,10)]
    with ProcessPoolExecutor(3,initializer=init) as workers:
        for j,_ in enumerate(workers.map(collect,jobs),1):
            if j%5==0: print("belief train",10*j,flush=True)
    tally=Counter()
    for p,_ in jobs:
        with np.load(p) as data:
            v,legal,old=data["viable"],data["ok"],data["default"]
            tally["observations"]+=len(old)
            tally["at_least_one_irreversible"]+=int((legal&~v).any(1).sum())
            tally["baseline_irreversible"]+=int((~v[np.arange(len(old)),old]).sum())
            tally["both"]+=int((data["phase"]==2).sum())
    save(HERE/f"collect-{n}.json",dict(train_sha256=digest(HERE.parent/"train.jsonl"),maps=[r["name"] for r in selected],
         weights=digest(HERE.parent/"runs/exit600/s900.pt"),source_sha256=digest(HERE/"collect.py"),summary=dict(tally),
         parts={p.name:digest(p) for p,_ in jobs}))
    print(tally,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("n",type=int)
    main(p.parse_args().n)
