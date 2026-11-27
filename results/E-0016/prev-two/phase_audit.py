"""Только train. Физический свидетель достижимости и разведка из одной точки передачи."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor
from heapq import heappop,heappush
from itertools import count
import statistics
import sys
import torch
from maze.environment import Environment,Status
from maze.grid import destination,locations
from .adapter import MOVES
from .frozen_adapter import load
from .common import HERE,digest,group,level,read,write

CTL=None


def initialize(tag):
    global CTL
    torch.set_num_threads(1)
    path=HERE/"weights"/(tag+".pt")
    CTL=load(path.read_bytes(),digest(path))


def restore(row,trace):
    env=Environment(level(row))
    prev=(None,None)
    for c in trace["prefix"]:
        prev=env.view.fuel_distance,env.view.repair_distance
        env.step(MOVES[int(c)])
    s=trace["snapshot"]
    assert list(prev)==s["previous"] and env.view.memory==tuple(s["memory"])
    return env,prev


def witness(env):
    grid=env.level.grid;goal=env.level.goal
    symbols={p:k for k in "FR" for p in locations(grid,k)}
    symbols[goal]="B"
    graph={}
    for origin in {*symbols,env.position}:
        paths={origin:""}; queue=deque([origin]);edges={}
        while queue:
            p=queue.popleft()
            for a,m in enumerate(MOVES):
                q=destination(grid,p,m)
                if q is None or q in paths:continue
                paths[q]=paths[p]+str(a)
                if q in symbols:edges[q]=paths[q]
                else:queue.append(q)
        graph[origin]=edges
    serial=count();queue=[(0,next(serial),env.position,env.fuel,env.health,"")];best={}
    while queue:
        length,_,p,f,h,route=heappop(queue)
        if p==goal:return route
        labels=best.setdefault(p,[])
        if any(of>=f and oh>=h for of,oh in labels):continue
        labels.append((f,h))
        for q,leg in graph[p].items():
            d=len(leg)
            if d>min(f,h) or env.steps+length+d>env.limit:continue
            nf=100 if symbols[q]=="F" else f-d
            nh=100 if symbols[q]=="R" else h-d
            if q==goal or min(nf,nh)>0:
                heappush(queue,(length+d,next(serial),q,nf,nh,route+leg))
    return None


def observe(env,actions=None,controller=None):
    cache={};cmd=[];seen=False;returns=0;same=0;last=env.position;initial=env.view
    while env.status==Status.RUNNING:
        v=env.view
        a=int(actions[len(cmd)]) if actions is not None else MOVES.index(controller.choose(v,cache))
        env.step(MOVES[a]);assert env.moved
        tile=env.view.memory[env.position[1]][env.position[0]]
        if tile in "FR":
            if seen and env.status==Status.RUNNING:
                returns+=1;same+=int(last==env.position)
            last=env.position;seen=False
        elif env.view.memory!=v.memory: seen=True
        cmd.append(str(a))
    if actions is not None: assert len(cmd)==len(actions)
    return dict(outcome=env.status.name,actions="".join(cmd),returns_after_discovery=returns,
                same_station_returns=same,discovered=sum(s.count("?") for s in initial.memory)-
                sum(s.count("?") for s in env.view.memory))


def work(job):
    row,held=job;trace=read(HERE/"takeovers"/(row["name"]+".json"))
    env,prev=restore(row,trace)
    proof=witness(env)
    if proof is not None:
        assert len(proof)==trace["shortest_finish"]
        for a in proof:env.step(MOVES[int(a)]);assert env.moved
        assert env.status==Status.SUCCESS
    else: assert trace["shortest_finish"] is None
    item=dict(witness=proof,held=held)
    for k in ("student","teacher"):
        env,_=restore(row,trace)
        item[k]=observe(env,trace[k]["actions"])
        assert item[k]["outcome"]==trace[k]["outcome"]
    path=HERE/"mandatory-train"/(row["name"]+".json")
    if path.exists():
        teacher=read(path);env,_=restore(row,trace)
        item["mandatory"]=observe(env,teacher["actions"])
        assert item["mandatory"]["outcome"]==teacher["outcome"]
    if held:
        env,previous=restore(row,trace)
        CTL.reset();CTL.previous=previous
        item["selected"]=observe(env,controller=CTL)
    return row["name"],item


def main(tag):
    info=read(HERE/"terminal/route-search.json");take=read(HERE/"takeovers.json")
    held=set(info["held_continuations"])
    batch=[(r,r["name"] in held) for r in group("extra")[:100]
           if read(HERE/"takeovers"/(r["name"]+".json"))["reached"]]
    with ProcessPoolExecutor(2,initializer=initialize,initargs=(tag,)) as workers:
        out=dict(workers.map(work,batch))
    summary={}
    for name in ("student","teacher","mandatory","selected"):
        subset=[r for r in out.values() if r["held"] and name in r]
        records=[r[name] for r in subset]
        if not records:continue
        summary[name]=dict(maps=len(records),wins=sum(r["outcome"]=="SUCCESS" for r in records),
             paired_gains=sum(r[name]["outcome"]=="SUCCESS" and r["student"]["outcome"]!="SUCCESS" for r in subset),
             paired_losses=sum(r[name]["outcome"]!="SUCCESS" and r["student"]["outcome"]=="SUCCESS" for r in subset),
             maps_with_return=sum(r["returns_after_discovery"]>0 for r in records),
             median_discovered=statistics.median(r["discovered"] for r in records),
             outcomes=dict(Counter(r["outcome"] for r in records)))
    write(HERE/"phase-audit.json",dict(selected=tag,held_maps=len(held),all_handoffs=len(out),
          physically_verified_witnesses=sum(r["witness"] is not None for r in out.values()),
          summary=summary,episodes=out))
    print("phase audit",len(held),summary,flush=True)


if __name__=="__main__":main(sys.argv[1])
