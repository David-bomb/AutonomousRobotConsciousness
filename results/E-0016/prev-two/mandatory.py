"""Применение зарегистрированного исторического учителя только к свежим состояниям train."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import import_module
import time
from maze.environment import Environment,Status
from maze.interface import Move
from .common import HERE,PARENT,SEED,digest,group,level,read,write

MOVES=tuple(Move)
Previous=import_module("results.verification-rl80.mandatory-proof-train.adapter").Previous


def play(row):
    path=HERE/"mandatory-train"/(row["name"]+".json")
    if path.exists():return read(path)
    trace=read(HERE/"takeovers"/(row["name"]+".json"))
    env=Environment(level(row));prev=(None,None)
    for a in trace["prefix"]:
        prev=env.view.fuel_distance,env.view.repair_distance
        env.step(MOVES[int(a)])
    assert list(prev)==trace["snapshot"]["previous"] and env.view.memory==tuple(trace["snapshot"]["memory"])
    teacher=Previous();teacher.previous=prev
    cache={};actions=[];start=time.perf_counter()
    while env.status==Status.RUNNING:
        v=env.view;key=v,teacher.previous
        if key in cache:
            move=cache[key];teacher.previous=(v.fuel_distance,v.repair_distance)
        else:
            move=teacher.choose(v);cache[key]=move
        assert teacher.previous==(v.fuel_distance,v.repair_distance)
        env.step(move)
        assert env.moved
        actions.append(str(MOVES.index(move)))
    out=dict(map=row["name"],geometry=row["geometry"],actions="".join(actions),outcome=env.status.name,
             unique_calls=len(cache),seconds=time.perf_counter()-start)
    write(path,out)
    return out


def main(workers=2):
    import numpy as np
    src=PARENT.parent/"verification-rl80/mandatory-proof-train"
    lock=read(src/"lock.json")["registration"]
    root=PARENT.parents[1]
    for name,h in lock["sources_sha256_lf"].items():
        assert sha256((root/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==h,name
    order=[row for row in group("extra")[:100] if read(HERE/"takeovers"/(row["name"]+".json"))["reached"]]
    rng=np.random.default_rng(SEED+202);rng.shuffle(order)
    held=set(read(HERE/"terminal/route-search.json")["held_continuations"])
    assert not ({r["name"] for r in order[:40]}&held)
    with ProcessPoolExecutor(workers) as pool:
        traces=list(pool.map(play,order[:8]))
        mean=sum(r["seconds"] for r in traces)/len(traces)
        chosen=order[:24]+order[40:52] if mean>25 else order
        print("mandatory probe",mean,"seconds/map;",len(chosen),"cases",flush=True)
        for out in pool.map(play,chosen[8:]):
            traces.append(out)
            if len(traces)%12==0:print("mandatory",len(traces),flush=True)
    split={}
    for label,subset in (("training",[r for r in traces if r["map"] not in held]),
                         ("held",[r for r in traces if r["map"] in held])):
        split[label]=dict(n=len(subset),wins=sum(r["outcome"]=="SUCCESS" for r in subset))
    write(HERE/"mandatory.json",dict(teacher="E-0010 mandatory-proof cold previous F/R",probe_seconds_per_map=mean,
          split=split,source_lock=digest(src/"lock.json"),adapter_sha256=digest(src/"adapter.py"),
          files={r["map"]:digest(HERE/"mandatory-train"/(r["map"]+".json")) for r in traces}))
    print("mandatory final",split,flush=True)


if __name__=="__main__":
    import sys
    main(*map(int,sys.argv[1:]))
