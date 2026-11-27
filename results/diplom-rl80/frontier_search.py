"""Learn how much to trust feasible finish and exploration routes in each physical action score."""
from hashlib import sha256
import json
import time

import numpy as np
import torch
from belief_search import EXP, HERE, WEIGHTS, Controller as SequenceController, base, data, fit_morphology, measure, sequence
from maze.grid import destination

OUT=HERE/"frontier-search"


def save(path,obj):
    payload=(json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=payload:raise ValueError(path)
    tmp=path.with_suffix(".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def rank_lengths(lengths, valid):
    if not valid.any():return np.zeros(8)
    diff=lengths-lengths[valid].min()
    out=np.zeros(8)
    out[valid]=np.exp(-np.maximum(diff[valid],0)*.60)
    return out


class FrontierController(SequenceController):
    def features(self,v):
        legal=np.asarray([destination(v.window,(1,1),m) is not None for m in base.MOVES])
        fields=np.zeros((8,9),np.float32)
        if legal.sum()<2:
            return np.where(legal,0.,-1.e6),fields
        x,checked=sequence.encode(v,self.prior)
        assert np.array_equal(legal,checked)
        with torch.inference_mode():
            logits=self.net(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))[0].numpy()
        logits=(logits-logits[legal].max())/max(float(logits[legal].std()),.5)
        # Five different stop orders share the same physical first action.
        plans=x[:,255:415].reshape(8,4,5,8)
        verified=plans[:,0]
        has=np.any(verified[:,:,6]>0,axis=1)&legal
        if has.any():
            lengths=np.where(verified[:,:,6]>0,verified[:,:,4]*75,10000).min(1)
            fields[:,0]=rank_lengths(lengths,has)
            return logits,fields
        survey=x[:,415:]
        absolute=survey[:,:12]*50
        exists=survey[:,24:]>0
        # Values are comparisons of candidate lengths, with physical feasibility
        # defined over the robot's known cell types and replenishment rules.
        for t in (2,1,0):
            ok=exists[:,4*t+1]&legal
            if not ok.any():continue
            corr=rank_lengths(absolute[:,4*t+1],ok)
            if t==2:
                fields[:,1]=corr
                fields[:,2]=rank_lengths(absolute[:,4*t+3],ok)
                return logits,fields
            slacks=(v.fuel-v.fuel_distance,v.health-v.repair_distance)
            if t==1:
                fields[:,3]=corr*(min(slacks)>5)
            if t==0:
                fields[:,4]=corr*(min(v.fuel,v.health)>40)
        slacks=(v.fuel-v.fuel_distance,v.health-v.repair_distance)
        index=int(slacks[1]<slacks[0])
        reading=(v.fuel_distance,v.repair_distance)[index]
        if min(slacks)<=15 and reading>0:
            world=self.est.features(v)
            fields[:,5]=world[:,index]*(min(slacks)<=3)
            fields[:,6]=world[:,index]*(3<min(slacks))
            fields[:,7]=world[:,3]
            fields[:,8]=world[:,4]
        return logits,fields


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    train=data.rows(EXP/"sequence/train.jsonl")
    prior=fit_morphology(data.rows(EXP/"data/train.jsonl")[:600])
    bot=FrontierController(prior)
    portfolio=np.asarray([
        [0,0,0,0,0,0,0,0,0], [12,8,0,0,0,0,0,0,0], [12,4,2,2,1,0,0,0,0],
        [15,10,1,2,0,4,1,2,1], [12,2,6,4,2,7,2,2,1], [15,8,0,4,4,3,0,2,0],
        [12,4,4,0,0,9,2,0,0], [10,0,0,0,0,5,1,0,0]],float)
    def batch(start,n):
        return [r for tier in ("H1","H2","H3","H4") for r in [z for z in train[start:start+100] if z["tier"]==tier][:n]]
    fit,select=batch(1300,8),batch(1400,15)
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    save(OUT/"protocol.json",dict(fit=[r["name"] for r in fit],select=[r["name"] for r in select],
         morphology=prior.tolist(),base_sha256=digest(WEIGHTS),source_sha256=digest(HERE/"frontier_search.py"),
         auxiliary={s:digest(HERE/s) for s in ("belief_worlds.dll","belief_search.py","belief_probe.py")},
         origin="All features derived from Observation. Ultimate action is the argmax of eight learned scores.",
         reward="one for terminal SUCCESS, zero otherwise",portfolio=portfolio.tolist(),
         rule="keep fit top three nonzero settings; select among zero, these three and their mean by training successes, earliest on ties; then one dev"))
    def run(name,rows,params,physical):
        path=OUT/name
        if path.exists():
            r=json.loads(path.read_bytes())
            assert r["coefficients"]==params.tolist() and r["maps"]==[x["name"] for x in rows]
            return np.asarray(r["wins"])
        t=time.monotonic()
        wins,episodes=measure(rows,params,bot,physical)
        save(path,dict(maps=[r["name"] for r in rows],coefficients=params.tolist(),wins=wins.tolist(),episodes=episodes,
                       seconds=round(time.monotonic()-t,2)))
        return wins
    fit_scores=run("train.json",fit,portfolio,False)
    best=portfolio[1+np.argsort(-fit_scores[1:],kind="stable")[:3]]
    params=np.vstack((portfolio[0],best,best.mean(0)))
    scores=run("select.json",select,params,False)
    selected=int(scores.argmax())
    save(OUT/"frozen.json",dict(coefficients=params[selected].tolist(),fit=fit_scores.tolist(),select=scores.tolist(),
                               selected=selected,protocol_sha256=digest(OUT/"protocol.json")))
    print("fit",fit_scores.tolist(),"select",scores.tolist(),flush=True)
    print("final dev",run("dev.physical.json",data.rows(EXP/"data/dev.jsonl"),params[selected:selected+1],True).tolist(),flush=True)


if __name__=="__main__":
    main()
