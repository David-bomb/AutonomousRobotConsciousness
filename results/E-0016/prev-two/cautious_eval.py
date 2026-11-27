"""Физический подбор допуска к замене, проверка на двух новых источниках и полный dev."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from .adapter import MOVES
from .cautious_bank import OUT,bank
from .cautious_fit import contrast
from .cautious_policy import evidence,load,urgency
from .common import DEV,HERE,digest,level,read,rows,write
from .early_policy import complete
from .early_bank import FOLDER
from .release import differences

THRESHOLDS=[2.,.90,.70,.50,.35,.20,.10,0.]
CTL=None


def init(path,abl):
    global CTL
    torch.set_num_threads(1)
    if path.parent.name=="geodesic":
        from .path_policy import load as factory
    else:factory=load
    CTL=factory(path.read_bytes(),abl)


def play(job):
    row,thresholds,mode=job
    env=Environment(level(row));cache={};sampled=set();measures={};trips=[]
    for index,threshold in enumerate(thresholds):
        env.reset();CTL.reset();actions=[];switch=None;both=None;edits=0
        while env.status==Status.RUNNING:
            v=env.view;pair=CTL.previous
            if switch is None and complete(v.memory):switch=env.steps
            if switch is not None and mode!="dev":break
            if mode=="dev":
                CTL.threshold=threshold
                move=CTL.choose(v,cache)
                assert CTL.previous==(v.fuel_distance,v.repair_distance)
            else:
                key=v,pair
                if key not in cache:
                    cache[key]=CTL.decision(v,(None,None) if CTL.ablate else pair)
                own,change,delta=cache[key]
                selected=change if delta>threshold else own
                edits+=int(selected!=own);move=MOVES[selected]
                if mode=="hold" and index==0 and key not in sampled:
                    k,slack=urgency(v)
                    if 0<=slack<=2:
                        legal=np.array([destination(v.window,(1,1),m) is not None for m in MOVES])
                        if legal.sum()>1:
                            sampled.add(key);signs=np.ones((8,2),int);curr=np.array((v.fuel_distance,v.repair_distance))
                            for a in np.flatnonzero(legal):
                                child=copy(env);child._memory=copy(env._memory);after=child.step(MOVES[a])
                                signs[a]=np.array((after.fuel_distance,after.repair_distance))-curr+1
                            stats=contrast(v,pair,evidence(v,pair),(None,legal),dict(urgent=k,signs=signs))
                            if hasattr(CTL,"walls"):
                                from .path_policy import diagnostics
                                stats.update(diagnostics(v,pair,CTL.walls,signs[:,k],k,legal))
                            for name,values in stats.items():measures.setdefault(name,Counter()).update(values)
                CTL.previous=v.fuel_distance,v.repair_distance
            env.step(move);assert env.moved;actions.append(str(MOVES.index(move)))
            if both is None and env.refuels and env.repairs:both=env.steps
        trips.append(dict(map=row["name"],geometry=row["geometry"],tier=row["tier"],handover=switch,
                both_at=both,steps=env.steps,edits=edits if mode!="dev" else None,outcome=env.status.name,
                actions="".join(actions)))
    return row["name"],trips,measures


def summarize(episodes):
    gates=[e["handover"] is not None for e in episodes]
    wins=[e["outcome"]=="SUCCESS" for e in episodes]
    by=lambda flags:[sum(ok and e["tier"]==f"H{i}" for e,ok in zip(episodes,flags)) for i in range(1,5)]
    return dict(n=len(episodes),wins=sum(wins),tiers=by(wins),reached=sum(gates),reached_tiers=by(gates),
        early_success=sum(w or g for w,g in zip(wins,gates)),edits=(sum(e["edits"] for e in episodes)
            if all(e["edits"] is not None for e in episodes) else None),
        both=sum(e["both_at"] is not None for e in episodes),failed_before=sum(not w and not g for w,g in zip(wins,gates)),
        failed_after=sum(not w and g for w,g in zip(wins,gates)) if all(e["outcome"]!="RUNNING" for e in episodes) else None)


def run(split,thresholds,path,ablate=False):
    data=rows(DEV) if split=="dev" else bank(split)
    records={};evidence_stats={}
    with ProcessPoolExecutor(2,initializer=init,initargs=(path,ablate)) as pool:
        for name,episodes,stats in pool.map(play,[(r,thresholds,split) for r in data]):
            records[name]=episodes
            for k,v in stats.items():evidence_stats.setdefault(k,Counter()).update(v)
            if len(records)%50==0:print("cautious",split,len(records),flush=True)
    by_model=[[trips[i] for trips in records.values()] for i in range(len(thresholds))]
    result=dict(sha256=digest(path),thresholds=thresholds,summaries=[summarize(x) for x in by_model],
                episodes=by_model,ablate=ablate,contrasts=evidence_stats)
    write(OUT/(split+("-ablated" if ablate else "")+".json"),result)
    print(split,[(s["reached"],s["wins"]) for s in result["summaries"]],flush=True)
    return result


def main():
    stage=OUT/"candidate.pt"
    sel=run("select",THRESHOLDS,stage)
    score=lambda s:(s["early_success"],sum(s["reached_tiers"][2:]),-s["edits"])
    winner=max(range(len(THRESHOLDS)),key=lambda i:score(sel["summaries"][i]))
    threshold=THRESHOLDS[winner]
    pack=torch.load(stage,map_location="cpu",weights_only=True);pack["threshold"]=threshold
    path=OUT/"frozen.pt";torch.save(pack,path)
    files=["cautious_policy.py","cautious_geometry.cpp","cautious_geometry.dll","early_policy.py","adapter.py"]
    if OUT.name=="geodesic":files.extend(("path_policy.py","path_counts.cpp","path_counts.dll"))
    write(OUT/"selection.json",dict(chosen=winner,threshold=threshold,sha256=digest(path),
          parent=digest(stage),summaries=sel["summaries"],sources={f:digest(HERE/f) for f in files}))
    hold=run("hold",[2.,threshold],path)
    run1=run("dev",[threshold],path);abl=run("dev",[threshold],path,True)
    base=read(FOLDER/"dev-wave1.json")
    honest=dict(episodes=run1["episodes"][0]);missing=dict(episodes=abl["episodes"][0])
    pairs=dict(versus_parent=differences(base,honest),versus_ablation=differences(missing,honest))
    for label,result in (("select",sel),("hold",hold)):
        a,b=result["episodes"][0],result["episodes"][winner if label=="select" else 1]
        def success(r):return r["handover"] is not None or r["outcome"]=="SUCCESS"
        pairs[label+"_early"]=dict(gains=sum(success(y) and not success(x) for x,y in zip(a,b)),
                                 losses=sum(success(x) and not success(y) for x,y in zip(a,b)))
    write(OUT/"paired.json",pairs)


if __name__=="__main__":
    if sys.argv[1:] == ["geodesic"]:
        from .path_bank import OUT,bank
    main()
