"""Поиск коэффициентов по физическому терминальному SUCCESS, с отбором только на train."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from maze.environment import Environment,Status
from ...data import OUT,level,rows,read,digest,save
from ...dev import evaluate
from ...policy import MOVES
from .features import HERE
from .physical_model import Model,load


def initialize(weights):
    global model
    torch.set_num_threads(1)
    model=Model(torch.load(weights,weights_only=True,map_location="cpu"),torch.from_numpy(np.load(HERE/"prior.npy")))


def trajectories(job):
    row,settings=job
    sim=level(row)
    cache={}
    outcomes=[]
    for weight in settings:
        w=np.array(weight)
        env=Environment(sim)
        moves={}
        while env.status==Status.RUNNING:
            v=env.view
            if v not in moves:
                if v not in cache:
                    cache[v]=model.directions(v)
                x,legal=cache[v]
                moves[v]=int(np.where(legal,x@w,-1e12).argmax())
            env.step(MOVES[moves[v]])
        outcomes.append(dict(outcome=env.status.name,steps=env.steps,both=bool(env.refuels and env.repairs)))
    return dict(map=row["name"],tier=row["tier"],episodes=outcomes)


def run(pool,bank,settings):
    results=[]
    for r in pool.map(trajectories,[(b,settings) for b in bank]):
        results.append(r)
        if len(results)%10==0:print("terminal search",len(results),len(bank),flush=True)
    score=[]
    for j,w in enumerate(settings):
        wins=Counter(r["tier"] for r in results if r["episodes"][j]["outcome"]=="SUCCESS")
        score.append(dict(weights=w,wins=sum(wins.values()),minimum=min(wins[t] for t in ("H1","H2","H3","H4"))))
    return dict(score=score,records=results)


def subset(bank,k):
    return [r for t in ("H1","H2","H3","H4") for r in [r for r in bank if r["tier"]==t][:k]]


def main(name):
    root=HERE.parent/"runs/exit600"
    path=root/(name+".pt")
    bank=rows(HERE.parent/"train.jsonl")
    out=HERE.parent/"runs"/("terminal-"+name)
    out.mkdir(exist_ok=False,parents=True)
    settings=[[1,0,0,0,0],[1,12,8,0,0]]
    settings.extend([1,12,s,early,late] for s in (4,8,16) for early,late in
                    ((4,0),(10,0),(0,5),(0,12),(8,5),(16,8)))
    config=dict(base=digest(path),prior=digest(HERE/"prior.npy"),survival_training=read(HERE/"prior.json"),
                screening_settings=settings,screen=[1400,1500,8],selection=[1500,1600,15],
                rank=["total SUCCESS","worst tier","earliest"],sources={n:digest(HERE/n) for n in (
                "population.py","physical_model.py","features.py","walks.cpp","walks.dll","physical_adapter.py")})
    save(out/"config.json",config)
    with ProcessPoolExecutor(2,initializer=initialize,initargs=(path,)) as pool:
        screen=run(pool,subset(bank[1400:1500],8),settings)
        save(out/"screen.json",screen)
        ranked=sorted(screen["score"],key=lambda s:(-s["wins"],-s["minimum"]))
        selected=[s["weights"] for s in ranked[:4]]
        for special in settings[:2]:
            if special not in selected:selected.append(special)
        selection=run(pool,subset(bank[1500:1600],15),selected)
        save(out/"train-selection.json",selection)
    record=max(selection["score"],key=lambda s:(s["wins"],s["minimum"]))
    pack=dict(base=torch.load(path,weights_only=True,map_location="cpu"),tables=torch.from_numpy(np.load(HERE/"prior.npy")),
              linear=torch.tensor(record["weights"],dtype=torch.float64))
    target=out/"rl.pt"
    torch.save(pack,target)
    sig=digest(target)
    save(out/"locked-before-dev.json",dict(sha256=sig,weights=record["weights"],train_selection=record,
         dev_sha256=digest(OUT/"data/dev.jsonl")))
    t=time.monotonic()
    summary,episodes=evaluate(load(target.read_bytes()),rows(OUT/"data/dev.jsonl"))
    save(target.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,seconds=time.monotonic()-t))
    print(record,summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("base")
    main(p.parse_args().base)
