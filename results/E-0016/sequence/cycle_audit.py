"""Состояния первого точно повторившегося наблюдения в учебном опыте с истинной меткой."""
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment
from ..data import digest,rows,read,level,save
from ..policy import MOVES,geometry
from ..teacher import Teacher
from .bank import HERE
from .model import encode


def main():
    source=HERE/"survey-teacher-100.json"
    bank={r["name"]:r for r in rows(HERE/"train.jsonl")}
    prior=torch.load(HERE/"runs/mission600/rl.pt",weights_only=True,map_location="cpu")["wall_prior"].numpy().tobytes()
    output=[]
    for r in read(source)["records"]:
        run=next(t for t in r["episodes"] if t["mode"]=="true_sensor")
        if run["outcome"]!="LIMIT":continue
        env=Environment(level(bank[r["map"]]))
        teacher=Teacher(env.level)
        seen={}
        seq=[]
        first=None
        for char in run["actions"]:
            v=env.view
            if v in seen:
                first=seen[v]
                break
            seen[v]=len(seq)
            seq.append((v,int(char)))
            env.step(MOVES[int(char)])
        if first is None:
            output.append(dict(map=r["map"],period=0))
            continue
        records=[]
        for v,a in seq[first:]:
            data,ok=encode(v,prior)
            valid=data[:,439:451].reshape(8,3,4)[:,:,0]>.5
            frontier=data[:,415:427].reshape(8,3,4)
            k=next((j for j in (2,1,0) if valid[:,j].any()),-1)
            minimum=frontier[:,k,1].min() if k<0 else frontier[valid[:,k],k,1].min()
            best=valid[:,k]&(np.abs(frontier[:,k,1]-minimum)<1e-6) if k>=0 else ok
            records.append(dict(available=k,chosen_can_reach=bool(k>=0 and valid[a,k]),shortest=bool(best[a]),
                                slack=min(v.health-v.repair_distance,v.fuel-v.fuel_distance),
                                known_cells=sum(len(line)-line.count("?") for line in v.memory)))
        output.append(dict(map=r["map"],first=first,period=len(records),visits=records))
    seen=[r for r in output if r["period"]]
    summary=dict(limit=len(output),cycles=len(seen),at_start=dict(Counter(r["visits"][0]["available"] for r in seen)),
        ever_both=sum(any(v["available"]==2 for v in r["visits"]) for r in seen),
        ever_one=sum(any(v["available"]==1 for v in r["visits"]) for r in seen),
        only_unprotected_or_none=sum(all(v["available"]<1 for v in r["visits"]) for r in seen),
        visits=dict(Counter(v["available"] for r in seen for v in r["visits"])))
    save(HERE/"cycle-audit.json",dict(source_sha256=digest(source),summary=summary,episodes=output))
    print(summary,flush=True)


if __name__=="__main__":
    main()
