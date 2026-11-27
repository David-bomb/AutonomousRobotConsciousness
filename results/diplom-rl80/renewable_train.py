"""Select learned exploration coefficients on full train returns and freeze before dev."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from renewable_policy import Controller,HERE

PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.environment import Environment,Status
from maze.interface import Move
data=import_module("results.E-0016.data")
OUT=HERE/"renewable-search"
EXP=PROJECT/"results/E-0016"
BASE=EXP/"prev-two/weights/wave1.pt"
BASE_HASH="369c0d5616a07a19f6fa992d53bdaa5a3004c5c54344c427f2345272b2f56432"
moves=tuple(Move)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    if path.exists():
        if json.loads(path.read_bytes())!=obj:raise ValueError(f"Inconsistent existing record {path}")
    else:data.save(path,obj)


def assess(rows,options,policy,physical=False,ablate=False):
    runs=[]
    wins=np.zeros(len(options),int)
    for i,r in enumerate(rows):
        env=Environment(data.level(r))
        decisions={}
        for p in range(len(options)):
            env.reset()
            old=(None,None)
            actions=bytearray()
            seen=set()
            while env.status==Status.RUNNING:
                v=env.view
                k=v,old
                if not physical:
                    if k in seen:break
                    seen.add(k)
                if k not in decisions:
                    q,x=policy.scores(v,old)
                    decisions[k]=(q[None,:]+options@x.T).argmax(1)
                a=int(decisions[k][p])
                actions.append(a)
                old=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                env.step(moves[a])
            outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
            wins[p]+=outcome=="SUCCESS"
            runs.append(dict(map=r["name"],tier=r["tier"],model=p,outcome=outcome,steps=env.steps,
                             refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest()))
        if (i+1)%10==0:print("maps",i+1,wins.tolist(),flush=True)
    return wins,runs


def main():
    torch.set_num_threads(1)
    assert digest(BASE)==BASE_HASH
    OUT.mkdir(exist_ok=True)
    corpus=HERE/"return-search/bank.jsonl"
    bank=data.rows(corpus)
    def balanced(rows,n):
        groups=[[r for r in rows if r["tier"]==t][-n:] for t in ("H1","H2","H3","H4")]
        return [row for same_index in zip(*groups,strict=True) for row in same_index]
    fit,select=balanced(bank[:100],8),balanced(bank[100:],14)
    options=np.array([[0,0,0,0,0,0,0],[18,0,12,0,0,0,0],[18,12,0,0,12,0,0],
                      [18,0,12,0,12,0,0],[18,0,0,12,0,12,0],[18,0,0,12,0,0,12],
                      [18,1,8,2,5,4,3],[18,4,0,8,3,3,4]],float)
    protocol=dict(base_sha256=BASE_HASH,bank_sha256=digest(corpus),fit=[r["name"] for r in fit],
                  select=[r["name"] for r in select],portfolio=options.tolist(),
                  rule="Terminal SUCCESS on full train maps. Top three nonzero vectors from fit, their mean and zero in separate full train select. Earliest on tie. Freeze; single rollout per dev map per ablation.",
                  features="Known full goal route; three frontier costs with proved repeatable resupply; three one-way forecasts after eligible safe observation runs out.",
                  inputs="Observation and two previous actual F/R readings only; empty at episode start",
                  source={name:digest(HERE/name) for name in ("renewable_features.cpp","renewable_features.dll","renewable_policy.py","renewable_train.py")})
    save(OUT/"protocol.json",protocol)
    controller=Controller(BASE.read_bytes())
    def run(name,rows,params,physical=False,ablate=False):
        path=OUT/name
        if path.exists():
            r=json.loads(path.read_bytes())
            assert r["weights"]==params.tolist() and r["maps"]==[r["name"] for r in rows]
            return np.array(r["wins"])
        start=time.monotonic()
        scores,episodes=assess(rows,params,controller,physical,ablate)
        obj=dict(maps=[r["name"] for r in rows],weights=params.tolist(),wins=scores.tolist(),episodes=episodes,
                 protocol_sha256=digest(OUT/"protocol.json"),seconds=round(time.monotonic()-start,2))
        if physical:
            obj.update(model_sha256=digest(OUT/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),
                       outcomes=dict(Counter(r["outcome"] for r in episodes)),ablated=ablate)
        save(path,obj)
        return scores
    scores=run("fit.json",fit,options)
    top=options[np.argsort(-scores[1:],kind="stable")[:3]+1]
    final=np.vstack((options[0],top,top.mean(0)))
    val=run("select.json",select,final)
    selected=final[int(val.argmax())]
    path=OUT/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp")
        torch.save(dict(format="renewable-1",core=torch.load(BASE,weights_only=True,map_location="cpu"),
                        weights=torch.tensor(selected,dtype=torch.float64)),tmp)
        tmp.replace(path)
    save(OUT/"frozen.json",dict(weights=selected.tolist(),base_sha256=BASE_HASH,model_sha256=digest(path),
                               fit=scores.tolist(),select=val.tolist(),select_sha256=digest(OUT/"select.json")))
    print("selection",val.tolist(),selected.tolist(),flush=True)
    dev=data.rows(EXP/"data/dev.jsonl")
    for tag,ablate in (("dev",False),("ablated",True)):
        print(tag,run(tag+".physical.json",dev,np.array([selected]),True,ablate).tolist(),flush=True)


if __name__=="__main__":main()
