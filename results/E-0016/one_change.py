"""Один альтернативный ход → дальнейшая самостоятельная поездка; только train."""
from copy import copy
from pathlib import Path
import argparse
from collections import Counter
from maze.environment import Environment,Status
from scripts.audit_automaton import Reachability
from .data import OUT,digest,rows,level,save
from .adapter import Controller
from .policy import MOVES


def drive(env,choose,trace=False):
    stamps=[]
    while env.status==Status.RUNNING:
        move=choose(env.view)
        if trace:
            stamps.append((env.position,env.fuel,env.health,move))
        env.step(move)
    return dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs),stamps


def main(weights):
    controller=Controller(weights,digest(weights))
    result=[]
    for row in rows(OUT/"data/train.jsonl")[700:800]:
        env=Environment(level(row))
        cached={}
        def choose(v):
            if v not in cached:
                cached[v]=controller.choose(v)
            return cached[v]
        base,stamps=drive(env,choose,True)
        record=dict(map=row["name"],tier=row["tier"],baseline=base)
        result.append(record)
        if env.status==Status.SUCCESS:
            continue
        oracle=Reachability(env.level)
        left,right=0,len(stamps)
        while right-left>1:
            mid=(right+left)//2
            pos,f,h,_=stamps[mid]
            if oracle.possible(pos,f,h,env.limit-mid):
                left=mid
            else:
                right=mid
        env.reset()
        for _,_,_,m in stamps[:left]:
            env.step(m)
        selected=stamps[left][-1]
        alternatives=[]
        for m,q in oracle.edges[env.position]:
            feasible=oracle.possible(q,*oracle.after(q,env.fuel,env.health),env.limit-left-1)
            if m==selected:
                assert not feasible
            if feasible:
                alt=copy(env)
                alt._memory=copy(env._memory)
                alt.step(m)
                finish,_=drive(alt,choose)
                alternatives.append(dict(move=m.name,**finish))
        assert alternatives
        record.update(step=left+1,selected=selected.name,alternatives=alternatives)
    opportunities=[r for r in result if "alternatives" in r]
    recovered=[r for r in opportunities if any(a["outcome"]=="SUCCESS" for a in r["alternatives"])]
    summary=dict(n=len(result),baseline_outcomes=dict(Counter(r["baseline"]["outcome"] for r in result)),
                 losses=len(opportunities),recovered_by_one=len(recovered),
                 alternate_episodes=sum(len(r["alternatives"]) for r in opportunities),
                 alternate_wins=sum(a["outcome"]=="SUCCESS" for r in opportunities for a in r["alternatives"]))
    save(weights.with_suffix(".one-change.json"),dict(weights_sha256=digest(weights),train_range=[700,800],
         train_sha256=digest(OUT/"data/train.jsonl"),summary=summary,episodes=result))
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("weights",type=Path)
    main(p.parse_args().weights)
