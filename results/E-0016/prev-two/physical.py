"""Полный ход с двумя предыдущими измерениями, в том числе при повторе ключа."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
import sys
import time
import torch
from maze.environment import Environment,Status
from maze.interface import Move
from ..sequence.adapter import load as load_base
from .common import BASE,BASE_SHA,DEV,HERE,SEQ,digest,level,read,register,rows,write

MOVES=tuple(Move)


def episode(row,ctl,mode,cache_choices=True):
    env=Environment(level(row))
    if mode!="baseline": ctl.reset()
    cache={}
    previous=(None,None)
    actions=[]
    prev_kind=None
    stats=Counter()
    stops=[None,None]
    while env.status==Status.RUNNING:
        v=env.view
        curr=(v.fuel_distance,v.repair_distance)
        if mode=="baseline":
            # Даже исходную сеть проверяем с полным ключом нового опыта.
            key=(v,previous)
            if cache_choices and key in cache:
                move=cache[key]
            else:
                move=ctl.choose(v)
                if cache_choices: cache[key]=move
        else:
            assert ctl.previous==previous
            move=ctl.choose(v,cache if cache_choices else None)
            assert ctl.previous==curr
        stocks=(v.fuel,v.health)
        margins=[stocks[k]-curr[k] for k in (0,1)]
        kind=min(range(2),key=margins.__getitem__)
        tight=curr[kind]>0 and margins[kind]<=3
        was_probe=tight and prev_kind==kind and previous[kind] is not None and curr[kind]>previous[kind]
        for k,z in enumerate(curr):
            if previous[k] is not None: assert abs(z-previous[k])<=1
        env.step(move)
        assert env.moved
        actual=(env.view.fuel_distance,env.view.repair_distance)
        if tight:
            stats["tight"]+=1
            stats["tight_down"]+=int(actual[kind]<curr[kind])
        if was_probe:
            stats["after_probe"]+=1
            stats["after_probe_down"]+=int(actual[kind]<curr[kind])
        if env.refuels and stops[0] is None: stops[0]=env.steps
        if env.repairs and stops[1] is None: stops[1]=env.steps
        actions.append(str(MOVES.index(move)))
        previous=curr
        prev_kind=kind if tight else None
    return dict(map=row["name"],geometry=row["geometry"],tier=row["tier"],steps=env.steps,
                outcome=env.status.name,refuels=env.refuels,repairs=env.repairs,first_f=stops[0],
                first_r=stops[1],actions="".join(actions),sensor=dict(stats))


def aggregate(trips):
    stages=Counter(); outcomes=Counter(); sensor=Counter()
    for e in trips:
        outcomes[e["outcome"]]+=1
        sensor.update(e["sensor"])
        if e["outcome"]!="SUCCESS":
            stages[int(bool(e["refuels"]))+int(bool(e["repairs"]))]+=1
    return dict(wins=outcomes["SUCCESS"],tiers=[sum(t["outcome"]=="SUCCESS" and t["tier"]==f"H{i}" for t in trips)
                                             for i in range(1,5)],outcomes=dict(outcomes),
                both=sum(bool(e["refuels"] and e["repairs"]) for e in trips),
                failed_by_services=[stages[i] for i in range(3)],sensor=dict(sensor),steps=sum(e["steps"] for e in trips))


def evaluate(tag,mode):
    reg=read(HERE/"registration.json")
    assert digest(DEV)==reg["dev_sha256"]
    if mode=="baseline":
        path=BASE; sha=BASE_SHA; ctl=load_base(path.read_bytes())
    else:
        path=HERE/"weights"/(tag+".pt")
        frozen=read(HERE/"weights"/(tag+".json"))
        assert digest(HERE/"adapter.py")==frozen["adapter_sha256"]
        if frozen.get("patch_adapter_sha256"):
            from .patch_adapter import Controller
            assert digest(HERE/"route_adapter.py")==frozen["route_adapter_sha256"]
            assert digest(HERE/"patch_adapter.py")==frozen["patch_adapter_sha256"]
        elif frozen.get("route_adapter_sha256"):
            from .route_adapter import Controller
            assert digest(HERE/"route_adapter.py")==frozen["route_adapter_sha256"]
        else:
            from .adapter import Controller
        sha=frozen["sha256"]
        ctl=Controller(path,sha,ablate=mode=="ablated")
    assert digest(path)==sha
    out=HERE/"dev"/(tag+"-"+mode+".json")
    if out.exists():
        data=read(out)
        assert data["weights_sha256"]==sha and data["dev_sha256"]==reg["dev_sha256"]
        return data["summary"]
    start=time.monotonic()
    trips=[]
    for row in rows(DEV): trips.append(episode(row,ctl,mode))
    assert len(trips)==len({r["geometry"] for r in trips})==100
    result=aggregate(trips)
    if mode=="baseline":
        reference=read(SEQ/"runs/mission600/rl.physical.json")
        assert all(all(t[k]==prev[k] for k in prev) for t,prev in zip(trips,reference["episodes"],strict=True))
    write(out,dict(weights_sha256=sha,dev_sha256=reg["dev_sha256"],mode=mode,summary=result,episodes=trips,
                   seconds=time.monotonic()-start))
    print(tag,mode,result,flush=True)
    return result


if __name__=="__main__":
    torch.set_num_threads(1)
    register()
    evaluate(sys.argv[1],sys.argv[2])
