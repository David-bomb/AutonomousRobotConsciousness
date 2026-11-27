"""Две физические поездки из одного самостоятельно достигнутого состояния train."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from heapq import heappop,heappush
from importlib import import_module
import json
import torch
from maze.environment import Environment,Status
from maze.frontier_reserve import ReserveFrontier
from maze.grid import locations,wave
from .adapter import Controller,MOVES
from .common import HERE,digest,group,level,read,write

POLICY=None
teacher_drive=import_module("results.E-0016.exploration-student.collect").drive


def init_worker():
    global POLICY
    torch.set_num_threads(1)
    f=HERE/"weights/super600.pt"
    POLICY=Controller(f,digest(f))


def exact_goal(env):
    # Рёбра ведут к ПЕРВОМУ встреченному сервису или B. Строятся только в аудиторе train.
    grid=env.level.grid;goal=env.level.goal;start=env.position
    stops={p:k for k in "FR" for p in locations(grid,k)}
    stops[goal]="B"
    graph={}
    for p in {*stops,start}:
        graph[p]=[(q,d) for q,d,_ in wave(grid,[p],stop_at=frozenset(stops)) if d and q in stops]
    queue=[(0,start,env.fuel,env.health)];seen={}
    while queue:
        length,p,f,h=heappop(queue)
        if p==goal:return length
        labels=seen.setdefault(p,[])
        if any(of>=f and oh>=h for of,oh in labels):continue
        labels.append((f,h))
        for q,d in graph[p]:
            if length+d>env.limit-env.steps or d>min(f,h):continue
            nf=100 if stops[q]=="F" else f-d
            nh=100 if stops[q]=="R" else h-d
            if q==goal or min(nf,nh)>0:heappush(queue,(length+d,q,nf,nh))
    return None


def snapshot(env,previous):
    v=env.view
    return dict(position=v.position,memory=v.memory,window=v.window,dx=v.dx,dy=v.dy,fuel=v.fuel,health=v.health,
                fuel_distance=v.fuel_distance,repair_distance=v.repair_distance,last_move=v.last_move.name,
                moved=v.moved,previous=list(previous),physical_steps=env.steps)


def play(row):
    out=HERE/"takeovers"/(row["name"]+".json")
    if out.exists():return read(out)
    env=Environment(level(row)); POLICY.reset()
    prefix=[];cache={}
    while env.status==Status.RUNNING and not (env.refuels and env.repairs):
        move=POLICY.choose(env.view,cache);env.step(move);prefix.append(str(MOVES.index(move)))
    result=dict(map=row["name"],tier=row["tier"],geometry=row["geometry"],prefix="".join(prefix))
    if env.status!=Status.RUNNING:
        result.update(reached=False,outcome=env.status.name)
    else:
        original=env.view; previous=POLICY.previous
        result.update(reached=True,snapshot=snapshot(env,previous),shortest_finish=exact_goal(env))
        route=[];initial_unknown=sum(s.count("?") for s in env.view.memory)
        while env.status==Status.RUNNING:
            move=POLICY.choose(env.view,cache);env.step(move);route.append(str(MOVES.index(move)))
        result["student"]=dict(actions="".join(route),outcome=env.status.name,
                               discovered=initial_unknown-sum(s.count("?") for s in env.view.memory))
        env.reset()
        for a in prefix:env.step(MOVES[int(a)])
        assert env.view==original
        teacher=teacher_drive(env,ReserveFrontier())
        result["teacher"]=dict(actions=teacher["actions"],outcome=env.status.name,
                               discovered=initial_unknown-sum(s.count("?") for s in env.view.memory))
        if result["shortest_finish"] is None:
            assert result["student"]["outcome"]!="SUCCESS" and env.status!=Status.SUCCESS
    write(out,result)
    return result


def main():
    batch=group("extra")[:100]
    with ProcessPoolExecutor(2,initializer=init_worker) as workers:
        result=[]
        for r in workers.map(play,batch):
            result.append(r)
            if len(result)%20==0:print("takeover",len(result),flush=True)
    reached=[r for r in result if r["reached"]]
    win=lambda r,k:r[k]["outcome"]=="SUCCESS"
    write(HERE/"takeovers.json",dict(parent="super600",weights_sha256=digest(HERE/"weights/super600.pt"),
          n=len(result),reached=len(reached),physically_solvable=sum(r["shortest_finish"] is not None for r in reached),
          original_wins=sum(win(r,"student") for r in reached),teacher_wins=sum(win(r,"teacher") for r in reached),
          teacher_gains=[r["map"] for r in reached if win(r,"teacher") and not win(r,"student")],
          teacher_losses=[r["map"] for r in reached if win(r,"student") and not win(r,"teacher")],
          files={r["map"]:digest(HERE/"takeovers"/(r["map"]+".json")) for r in result}))


if __name__=="__main__":main()
