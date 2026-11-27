"""Measure action-conditional returns on a map-disjoint corpus of training forks."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import import_module
import numpy as np
import torch
from renewable_train import EXP,HERE,Environment,Status,data,digest,moves,save
from handover_train import complete
from rollout_policy import BranchNet
from scripts.audit_automaton import Reachability

OUT=HERE/"rollout-search"
BASE=HERE/"service-window/fixed/model.pt"
fork=import_module("results.E-0016.sequence.branches").fork


def initialize():
    global net
    torch.set_num_threads(1)
    net=BranchNet(torch.load(BASE,weights_only=True,map_location="cpu"))


def collect(row):
    name=row["name"];folder=OUT/"branches";packed=folder/(name+".npz");path=folder/(name+".json")
    if path.exists():
        report=data.read(path)
        assert digest(packed)==report["features"]
        return report["samples"]
    env=Environment(data.level(row));records=[];seen=set();cache={}
    rng=np.random.default_rng(int(row["geometry"][:12],16))
    past=(None,None)
    def inspect(v,old):
        key=v,old
        if key not in cache:
            q,_,legal,info=net.input(v,old,False);cache[key]=int(q.argmax()),legal,info
        return cache[key]
    while env.status==Status.RUNNING and not net.route(env.view,net.cfg):
        v=env.view;k=v,past
        if k in seen:break
        seen.add(k)
        a,legal,info=inspect(v,past)
        records.append((fork(env),past,info))
        past=(v.fuel_distance,v.repair_distance);env.step(moves[a])
    points=[i for i,(_,_,info) in enumerate(records) if info is not None]
    picks=set(points[:1])
    for lo,hi in ((-100,2),(3,10)):
        options=[i for i in points if lo<=records[i][2][1]<=hi]
        if options:picks.add(int(rng.choice(options)))
    if points and env.status in (Status.EMPTY,Status.BROKEN):
        oracle=Reachability(env.level)
        low,high=0,len(records)
        while high-low>1:
            mid=(high+low)//2;s=records[mid][0]
            if oracle.possible(s.position,s.fuel,s.health,s.limit-s.steps):low=mid
            else:high=mid
        for sign in (-1,1):
            near=[i for i in points if (i-low)*sign>=0]
            if near:picks.add(min(near,key=lambda i:abs(i-low)))
    values={};late={};descriptions=[];inputs=[];labels=[];masks=[];parents=[]
    for i in sorted(picks):
        state,old,info=records[i];v=state.view
        q,x,legal,_=net.input(v,old);effects=np.zeros((8,3),np.uint8)
        real=[]
        for a in np.flatnonzero(legal):
            child=fork(state);child.step(moves[a])
            before=(v.fuel_distance,v.repair_distance)
            urgent=info[0];after=(child.view.fuel_distance,child.view.repair_distance)
            descent=after[urgent]<before[urgent]
            visited=set()
            while child.status==Status.RUNNING and not net.route(child.view,net.cfg):
                nxt=child.view;key=nxt,before
                if key in visited:break
                visited.add(key)
                chosen=inspect(nxt,before)[0];before=nxt.fuel_distance,nxt.repair_distance;child.step(moves[chosen])
            hit=child.status==Status.RUNNING and net.route(child.view,net.cfg)
            if hit:ending=complete(child,net,late,values)[0]
            else:ending=child.status.name if child.status!=Status.RUNNING else "exact_cycle"
            effects[a]=[ending=="SUCCESS",hit or ending=="SUCCESS",descent]
            real.append([int(a),ending,bool(hit),bool(descent)])
        inputs.append(x);labels.append(effects);masks.append(legal);parents.append(int(q.argmax()))
        descriptions.append(dict(step=state.steps,slack=info[1],parent=int(q.argmax()),responses=real))
    payload={"x":np.array(inputs,dtype=np.float16),"y":np.array(labels,dtype=np.uint8),
             "legal":np.array(masks,dtype=bool),"parent":np.array(parents,dtype=np.int8)}
    temp=packed.with_suffix(".pending")
    with temp.open("wb") as f:np.savez_compressed(f,**payload)
    temp.replace(packed)
    save(path,dict(map=name,tier=row["tier"],geometry=row["geometry"],samples=len(inputs),features=digest(packed),
                   base=digest(BASE),forks=descriptions))
    return len(inputs)


def main():
    OUT.mkdir(exist_ok=True);(OUT/"branches").mkdir(exist_ok=True)
    bank=EXP/"sequence/spatial/large.jsonl";all_rows=data.rows(bank)
    # The source bank is training-only. Use unique, untouched slices for fitting and model checks.
    fit=all_rows[400:750];held=all_rows[1100:1200]
    assert len({r["geometry"] for r in fit+held})==len(fit)+len(held)
    save(OUT/"collection.json",dict(source=digest(bank),base=digest(BASE),fit=[r["name"] for r in fit],
        held=[r["name"] for r in held],code={n:digest(HERE/n) for n in ("rollout_branches.py","rollout_policy.py",
               "handover_train.py","tree_paths.dll","geodesic_evidence.dll")},
        rule="Own ordinary prefix of frozen wave1. Choose first ambiguous fork, one additional random fork per slack band, and adjacent ambiguous forks around the first true irreversible error on failed training trips. For each legal action replay the frozen controller, with the next two inputs taken only from actual steps, to its two service windows and through joint400 to its terminal outcome. All labels come from training maps."))
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        total=0
        for i,n in enumerate(pool.map(collect,fit+held)):
            total+=n
            if (i+1)%30==0:print(i+1,total,"train forks",flush=True)
    print("Complete",total,flush=True)


if __name__=="__main__":main()
