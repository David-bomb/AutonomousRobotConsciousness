"""Train a terminal-reward action head with the two real readings explicitly authorized by the user."""
from collections import Counter
from hashlib import sha256
import json
import time

import numpy as np
import torch
from belief_search import EXP, HERE, base, data, fit_morphology, sequence
from belief_probe import GeometryBelief
from frontier_search import FrontierController, save
from maze.environment import Environment, Status
from maze.grid import destination

OUT=HERE/"two-readings-search"
WEIGHTS=EXP/"sequence/runs/mission600/rl.pt"


class TwoReadingsPolicy(FrontierController):
    def __init__(self,priors):
        state=torch.load(WEIGHTS,map_location="cpu",weights_only=True)
        self.net=sequence.SequenceNet("plain").eval()
        self.net.load_state_dict(state)
        self.prior=self.net.wall_prior.numpy().tobytes()
        self.est=GeometryBelief(np.asarray(priors,np.float64),48)

    @staticmethod
    def history_fields(v, previous, spatial):
        fields=np.zeros((8,5),np.float32)
        if previous is None or not v.moved:
            return fields
        current=(v.fuel_distance,v.repair_distance)
        slack=(v.fuel-current[0],v.health-current[1])
        i=int(slack[1]<slack[0])
        if current[i]==0 or slack[i]>25:
            return fields
        dx,dy=v.last_move.value
        prior=v.position[0]-dx,v.position[1]-dy
        certain_return=previous[i]<current[i]
        # If the previous actual reading equalled D-1, reversing this move lowers
        # that sensor by one. The competing resource may make this undesirable.
        reverse=base.MOVES.index(type(v.last_move)((-dx,-dy)))
        verified_plan=bool(np.any(spatial[:,:3]))
        if certain_return:
            if verified_plan:
                fields[reverse,4]=1
            elif slack[i]<=3:
                fields[reverse,0]=1
            elif slack[i]<=10:
                fields[reverse,1]=1
            else:
                fields[reverse,2]=1
        if not verified_plan and slack[i]<=12 and previous[i]>=current[i]:
            previous_neighbors={q for m in base.MOVES if (q:=destination(v.memory,prior,m)) is not None}
            for a,m in enumerate(base.MOVES):
                q=destination(v.memory,v.position,m)
                if q==prior or (previous[i]>current[i] and q in previous_neighbors):
                    fields[a,3]=-1
        return fields


def trials(cases,policies,net,physical=False,ablate=False):
    totals=np.zeros(len(policies),int)
    episodes=[]
    for k,row in enumerate(cases):
        env=Environment(data.level(row))
        geometry,actions={},{}
        for idx in range(len(policies)):
            env.reset()
            previous=None
            visited=set()
            while env.status==Status.RUNNING:
                v=env.view
                key=(v,previous)
                if not physical:
                    if key in visited:break
                    visited.add(key)
                if key not in actions:
                    if v not in geometry:geometry[v]=net.features(v)
                    q,spatial=geometry[v]
                    history=net.history_fields(v,previous,spatial)
                    f=np.concatenate((spatial,history),axis=1)
                    actions[key]=(q[None,:]+policies@f.T).argmax(1)
                command=base.MOVES[actions[key][idx]]
                previous=None if ablate else (v.fuel_distance,v.repair_distance)
                env.step(command)
            status=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
            totals[idx]+=status=="SUCCESS"
            episodes.append(dict(map=row["name"],tier=row["tier"],model=idx,outcome=status,steps=env.steps,
                                 refuels=env.refuels,repairs=env.repairs))
        if (k+1)%20==0:
            print("cases",k+1,totals.tolist(),flush=True)
    return totals,episodes


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    morphology=fit_morphology(data.rows(EXP/"data/train.jsonl")[:600])
    model=TwoReadingsPolicy(morphology)
    train=data.rows(EXP/"sequence/train.jsonl")
    def balanced(start,n):
        return [r for t in ("H1","H2","H3","H4") for r in [c for c in train[start:start+100] if c["tier"]==t][:n]]
    fit,select=balanced(900,8),balanced(1500,15)
    heads=[
        [0]*14,
        [0]*9+[8,3,0,5,0],
        [12,8,0,0,0,0,0,0,0,8,3,0,5,0],
        [15,10,1,2,0,4,1,2,1,12,5,1,6,0],
        [12,2,6,4,2,7,2,2,1,12,5,1,6,0],
        [14,8,0,2,2,4,1,2,1,16,10,2,8,1],
        [14,8,0,2,2,4,1,2,1,8,2,0,8,0],
        [12,6,2,0,0,3,0,1,0,16,8,0,4,4]]
    portfolio=np.asarray(heads,float)
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    save(OUT/"protocol.json",dict(fit=[r["name"] for r in fit],selection=[r["name"] for r in select],
         morphology=morphology.tolist(),portfolio=portfolio.tolist(),base_sha256=digest(WEIGHTS),source_sha256={f:digest(HERE/f)
         for f in ("two_readings_search.py","frontier_search.py","belief_search.py","belief_probe.py","belief_worlds.dll")},
         allowed="exactly two previous real F/R distance readings; cleared on reset, updated on each step including cache hits",
         validation="frozen full episodes; physical env.step; cache key includes previous actual pair",
         reward="terminal SUCCESS",rule="best three nonzero portfolios on fit; compare them, mean and zero on distinct selection; earliest on ties",
         ablation="selected frozen model with both prior readings missing at every step"))
    def measure(name,cases,params,physical=False,ablate=False):
        path=OUT/name
        if path.exists():
            saved=json.loads(path.read_bytes())
            assert saved["coefficients"]==params.tolist() and saved["maps"]==[r["name"] for r in cases]
            return np.asarray(saved["wins"])
        start=time.monotonic()
        wins,records=trials(cases,params,model,physical,ablate)
        save(path,dict(maps=[r["name"] for r in cases],coefficients=params.tolist(),wins=wins.tolist(),
                       episodes=records,seconds=round(time.monotonic()-start,2)))
        return wins
    fit_scores=measure("fit.json",fit,portfolio)
    best=portfolio[np.argsort(-fit_scores[1:],kind="stable")[:3]+1]
    options=np.vstack((portfolio[0],best,best.mean(axis=0)))
    selection=measure("select.json",select,options)
    chosen=int(selection.argmax())
    save(OUT/"frozen.json",dict(coefficients=options[chosen].tolist(),morphology=morphology.tolist(),
                               base_sha256=digest(WEIGHTS),protocol_sha256=digest(OUT/"protocol.json"),
                               fit=fit_scores.tolist(),selection=selection.tolist(),chosen=chosen))
    print("frozen",fit_scores.tolist(),selection.tolist(),flush=True)
    dev=data.rows(EXP/"data/dev.jsonl")
    winner=options[chosen:chosen+1]
    a=measure("dev.physical.json",dev,winner,True)
    b=measure("ablated.physical.json",dev,winner,True,True)
    print("real",a.tolist(),"ablated",b.tolist(),flush=True)


if __name__=="__main__":
    main()
