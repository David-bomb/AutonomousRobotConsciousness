"""Парный отбор весов восьмиканального последнего слоя по терминальным поездкам train."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse
import time
import numpy as np
import torch
from maze.environment import Environment,Status
from ...data import OUT,rows,read,level,save,digest
from ...dev import evaluate
from ...policy import MOVES
from .features import HERE
from .joint_model import Joint,load
from .population import subset


def package():
    base=HERE.parent/"runs/exit600/s900.pt"
    safety=HERE/f"safety-{read(HERE/'safety.json')['selected']}.pt"
    return dict(base=torch.load(base,weights_only=True,map_location="cpu"),
           safety=torch.load(safety,weights_only=True,map_location="cpu"),tables=torch.from_numpy(np.load(HERE/"prior.npy")))


def init():
    global net
    torch.set_num_threads(1)
    net=Joint(package())


def trial(job):
    row,weights=job
    e=Environment(level(row))
    cache={}
    records=[]
    for w in weights:
        e.reset()
        chosen={}
        while e.status==Status.RUNNING:
            v=e.view
            if v not in chosen:
                if v not in cache:cache[v]=net.directions(v)
                x,ok=cache[v]
                chosen[v]=int(np.where(ok,x@w,-1e12).argmax())
            e.step(MOVES[chosen[v]])
        records.append(dict(outcome=e.status.name,steps=e.steps,refuels=e.refuels,repairs=e.repairs))
    return dict(map=row["name"],tier=row["tier"],runs=records)


def sweep(pool,bank,weights):
    records=[]
    for r in pool.map(trial,[(r,weights) for r in bank]):
        records.append(r)
        if len(records)%12==0:print("joint",len(records),len(bank),flush=True)
    scores=[]
    for j,w in enumerate(weights):
        wins=Counter(r["tier"] for r in records if r["runs"][j]["outcome"]=="SUCCESS")
        scores.append(dict(weights=w,wins=sum(wins.values()),minimum=min(wins[t] for t in ("H1","H2","H3","H4"))))
    return dict(scores=scores,records=records)


def main():
    # Первая часть — одинаковые физические старты при разных весах, вторая —
    # другие геометрии ещё до проверки на dev. Группы стартов сбалансированы по H.
    train=rows(HERE.parent/"train.jsonl")
    out=HERE.parent/"runs/joint400"
    out.mkdir(exist_ok=False,parents=True)
    configurations=[[1,0,0,0,0,0,0,0,0,0],[1,12,8,0,0,0,0,0,0,0]]
    for late,early in (([0,0,0,0],True),([8,6,0,0],True),([6,9,0,0],True),
                       ([8,8,4,2],True),([12,8,0,0],True),([8,6,0,0],False)):
        choices=([2,4,4],[4,8,8],[8,16,12],[4,4,16]) if early else ([0,0,0],)
        for safety in choices:configurations.append([1,12,8,*late,*safety])
    save(out/"config.json",dict(search=configurations,screen=[500,600,9],selection=[1300,1400,18],
         safety_weights=read(HERE/"safety.json")["selected"],order="wins, minimum tier, earliest",
         source_sha256={n:digest(HERE/n) for n in ("joint_train.py","joint_model.py","safety_net.py","itineraries.cpp",
                "itineraries.dll","walks.cpp","walks.dll","features.py")}))
    with ProcessPoolExecutor(2,initializer=init) as pool:
        screen=sweep(pool,subset(train[500:600],9),configurations)
        save(out/"screen.json",screen)
        ranked=sorted(screen["scores"],key=lambda r:(-r["wins"],-r["minimum"]))
        short=[r["weights"] for r in ranked[:5]]
        for w in configurations[:2]:
            if w not in short:short.append(w)
        selected=sweep(pool,subset(train[1300:1400],18),short)
        save(out/"selection.json",selected)
    best=max(selected["scores"],key=lambda r:(r["wins"],r["minimum"]))
    pack=package()
    pack["linear"]=torch.tensor(best["weights"],dtype=torch.float64)
    path=out/"rl.pt"
    torch.save(pack,path)
    sha=digest(path)
    save(out/"locked-before-dev.json",dict(sha256=sha,train=best,dev_sha256=digest(OUT/"data/dev.jsonl")))
    t=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),rows(OUT/"data/dev.jsonl"))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sha,dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,seconds=time.monotonic()-t))
    print(best,summary,flush=True)


if __name__=="__main__":
    main()
