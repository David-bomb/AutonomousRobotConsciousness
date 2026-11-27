"""Fit a small action-value head by terminal reward using latent geometries and sequence logits."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from belief_probe import HERE, PROJECT, GeometryBelief, base, data, fit_morphology, import_module
from maze.environment import Environment, Status
from maze.grid import destination

EXP=PROJECT/"results/E-0016"
OUT=HERE/"belief-search"
WEIGHTS=EXP/"sequence/runs/mission600/s8.pt"
sequence=import_module("results.E-0016.sequence.model")


def archive(name, obj):
    target=OUT/name
    raw=(json.dumps(obj,indent=2,ensure_ascii=False)+"\n").encode()
    if target.exists() and target.read_bytes()!=raw:
        raise ValueError(f"Existing record differs: {target}")
    temp=target.with_suffix(".tmp")
    temp.write_bytes(raw)
    temp.replace(target)


class Controller:
    def __init__(self, morphology):
        state=torch.load(WEIGHTS, weights_only=True, map_location="cpu")
        self.net=sequence.SequenceNet("plain").eval()
        self.net.load_state_dict(state)
        self.prior=self.net.wall_prior.numpy().tobytes()
        self.est=GeometryBelief(np.asarray(morphology,np.float64),48)

    def features(self,v):
        legal=np.asarray([destination(v.window,(1,1),m) is not None for m in base.MOVES])
        fields=np.zeros((8,5),np.float32)
        if legal.sum()<2:
            return np.where(legal,0.,-1.e6),fields
        x,checked=sequence.encode(v,self.prior)
        assert np.array_equal(legal,checked)
        with torch.inference_mode():
            logits=self.net(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))[0].numpy()
        scale=max(float(logits[legal].std()),.5)
        logits=(logits-logits[legal].max())/scale
        slacks=(v.fuel-v.fuel_distance,v.health-v.repair_distance)
        urgent=int(slacks[1]<slacks[0])
        slack=slacks[urgent]
        if slack>15:
            return logits,fields
        sensor=(v.fuel_distance,v.repair_distance)[urgent]
        if sensor==0:
            return logits,fields
        world=self.est.features(v)
        fields[:,0]=world[:,urgent]*(slack<=2)
        fields[:,1]=world[:,urgent]*(2<slack<=7)
        fields[:,2]=world[:,urgent]*(7<slack)
        fields[:,3]=world[:,3]
        fields[:,4]=world[:,4]
        return logits,fields


def measure(cases, params, net, physical):
    scores=np.zeros(len(params),int)
    records=[]
    for irow,row in enumerate(cases):
        env=Environment(data.level(row))
        policies={}
        for j in range(len(params)):
            env.reset()
            seen=set()
            while env.status==Status.RUNNING:
                v=env.view
                if not physical:
                    if v in seen:break
                    seen.add(v)
                if v not in policies:
                    base_q,features=net.features(v)
                    policies[v]=(base_q[None,:]+params@features.T).argmax(1)
                env.step(base.MOVES[policies[v][j]])
            outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
            scores[j]+=outcome=="SUCCESS"
            records.append(dict(map=row["name"],tier=row["tier"],model=j,outcome=outcome,
                                steps=env.steps,refuels=env.refuels,repairs=env.repairs))
        if (irow+1)%20==0: print(irow+1,"maps",scores.tolist(),flush=True)
    return scores,records


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    extra=data.rows(EXP/"sequence/train.jsonl")
    # Four fresh sub-batches, each with the same number of H1..H4.
    def balanced(start,n):
        return [row for tier in ("H1","H2","H3","H4")
                for row in [r for r in extra[start:start+100] if r["tier"]==tier][:n]]
    train,select=balanced(1100,8),balanced(1200,12)
    prior=fit_morphology(data.rows(EXP/"data/train.jsonl")[:600])
    controller=Controller(prior)
    portfolio=np.asarray([[0,0,0,0,0],[2,.5,0,.5,.5],[5,2,.5,1,1],[10,3,1,2,.5],
                          [6,3,1,3,3],[.5,0,0,4,4],[4,0,0,1,0],[12,0,0,0,0]],np.float64)
    sig=lambda p:sha256(p.read_bytes()).hexdigest()
    archive("protocol.json",dict(base_sha256=sig(WEIGHTS),source_sha256={s:sig(HERE/s) for s in
            ("belief_worlds.cpp","belief_worlds.dll","belief_search.py","belief_probe.py")},
            morphology=prior.tolist(),particles=48,fit_maps=[r["name"] for r in train],
            select_maps=[r["name"] for r in select],portfolio=portfolio.tolist(),reward="terminal SUCCESS",
            rule="best two nonzero fit settings, then compare them, their midpoint, and zero on separate train select; earliest on ties"))
    def run(name,cases,variants,physical):
        path=OUT/name
        if path.exists():
            r=json.loads(path.read_bytes())
            assert r["coefficients"]==variants.tolist() and r["maps"]==[c["name"] for c in cases]
            return np.asarray(r["wins"])
        t=time.monotonic()
        wins,episodes=measure(cases,variants,controller,physical)
        archive(name,dict(coefficients=variants.tolist(),maps=[c["name"] for c in cases],wins=wins.tolist(),
                          episodes=episodes,seconds=round(time.monotonic()-t,2)))
        return wins
    scores=run("train.json",train,portfolio,False)
    i,j=np.argsort(-scores[1:],kind="stable")[:2]+1
    final=np.stack((portfolio[0],portfolio[i],portfolio[j],(portfolio[i]+portfolio[j])/2))
    selected=run("select.json",select,final,False)
    index=int(selected.argmax())
    archive("frozen.json",dict(base_sha256=sig(WEIGHTS),weights=final[index].tolist(),morphology=prior.tolist(),
                               fit_success=scores.tolist(),selection_success=selected.tolist(),selected=index,
                               protocol_sha256=sig(OUT/"protocol.json")))
    print("train",scores.tolist(),"select",selected.tolist(),"chosen",index,flush=True)
    win=run("dev.physical.json",data.rows(EXP/"data/dev.jsonl"),final[index:index+1],True)
    print("dev",win.tolist(),flush=True)


if __name__=="__main__":
    main()
