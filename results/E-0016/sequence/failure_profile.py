"""Полные поездки на последней сотне первичного train; истинная карта только у разбора."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from scripts.audit_automaton import Reachability
from ..adapter import from_bytes
from ..data import OUT,digest,level,rows,save
from ..policy import MOVES
from .adapter import load


def main(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    net=(load if "base.site.0.weight" in state else from_bytes)(path.read_bytes())
    results=[]
    for row in rows(OUT/"data/train.jsonl")[700:800]:
        env=Environment(level(row))
        cache={}
        trace=[]
        dead_end_steps=0
        retreat=0
        stalled=longest=0
        prev=None
        while env.status==Status.RUNNING:
            v=env.view
            if v not in cache:
                cache[v]=net.choose(v)
            a=cache[v]
            degree=sum(destination(v.window,(1,1),m) is not None for m in MOVES)
            dead_end_steps+=int(degree==1)
            trace.append((env.position,env.fuel,env.health,env.refuels,env.repairs,degree))
            env.step(a)
            retreat+=int(env.position==prev)
            prev=v.position
            stalled=stalled+1 if env.view.memory==v.memory else 0
            longest=max(stalled,longest)
        entry=dict(map=row["name"],outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                   end_degree=sum(destination(env.view.window,(1,1),m) is not None for m in MOVES),
                   dead_end_steps=dead_end_steps,immediate_returns=retreat,longest_without_new_cells=longest)
        results.append(entry)
        if env.status==Status.SUCCESS:
            continue
        oracle=Reachability(env.level)
        lo,hi=0,len(trace)
        while hi-lo>1:
            mid=(lo+hi)//2
            if oracle.possible(*trace[mid][:3],env.limit-mid):
                lo=mid
            else:
                hi=mid
        pos,f,h,fu,re,degree=trace[lo]
        entry.update(loss_step=lo+1,loss_stage="both" if fu and re else "one" if fu or re else "before",
                     loss_degree=degree)
    failed=[e for e in results if "loss_step" in e]
    summary=dict(outcomes=dict(Counter(r["outcome"] for r in results)),
                 loss_stages=dict(Counter(e["loss_stage"] for e in failed)),
                 median_first_loss=median(e["loss_step"] for e in failed) if failed else None,
                 losses_at_dead_end=sum(e["loss_degree"]==1 for e in failed),
                 stops_at_dead_end=sum(e["end_degree"]==1 for e in failed),
                 stalls_at_least_20=sum(e["longest_without_new_cells"]>=20 for e in failed))
    save(path.with_suffix(".failure-profile.json"),dict(weights_sha256=digest(path),train_range=[700,800],
         sample_sha256=digest(OUT/"data/train.jsonl"),summary=summary,episodes=results))
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("weights",type=Path)
    main(parser.parse_args().weights)
