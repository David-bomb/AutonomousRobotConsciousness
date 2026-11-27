"""Train the action weights first on post-service continuations, then select on full fresh train starts."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
import time
import numpy as np
import torch
from return_policy import HERE, PROJECT, ReturnPolicy, WEIGHTS, WEIGHTS_HASH, base, prev
from maze.environment import Environment, Status
from maze.datasets import geometry

EXP=PROJECT/"results/E-0016"
OUT=HERE/"return-search"
data=import_module("results.E-0016.data")


def save(path,obj):
    encoded=(json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=encoded:raise ValueError(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(encoded)
    tmp.replace(path)


def make_bank():
    path=OUT/"bank.jsonl"
    if path.exists():return data.rows(path)
    used=data.excluded()
    for source in (EXP/"data/train.jsonl",EXP/"data/dev.jsonl",EXP/"data/confirmation.jsonl",
                   EXP/"sequence/train.jsonl",EXP/"sequence/probe.jsonl"):
        used.update(r["geometry"] for r in data.rows(source))
    rows=[]
    for seed in (9165200,9165201):
        for r in data.generate(("return-train",seed)):
            h=geometry(r["grid"])
            assert h==r["geometry"] and h not in used
            used.add(h)
            rows.append(r)
    tmp=path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows),encoding="utf8")
    tmp.replace(path)
    return rows


def prefixes(rows):
    controller=prev.Controller(WEIGHTS,WEIGHTS_HASH)
    records=[]
    for row in rows:
        env=Environment(data.level(row))
        controller.reset()
        actions=[]
        seen=set()
        while env.status==Status.RUNNING and not (env.refuels and env.repairs):
            key=env.view,controller.previous
            if key in seen:break
            seen.add(key)
            m=controller.choose(env.view)
            actions.append(base.MOVES.index(m))
            env.step(m)
        records.append(dict(map=row["name"],reached=bool(env.refuels and env.repairs and env.status==Status.RUNNING),
                            actions=actions,steps=env.steps))
    return records


def series(cases,params,net,suffixes=None,physical=False,ablate=False):
    wins=np.zeros(len(params),int)
    records=[]
    for k,row in enumerate(cases):
        prefix=[] if suffixes is None else suffixes[row["name"]]["actions"]
        if suffixes is not None and not suffixes[row["name"]]["reached"]:continue
        env=Environment(data.level(row))
        observed,decisions={},{}
        for model in range(len(params)):
            env.reset()
            prior=(None,None)
            for a in prefix:
                prior=(env.view.fuel_distance,env.view.repair_distance)
                env.step(base.MOVES[a])
            seen=set()
            actions=bytearray(prefix)
            while env.status==Status.RUNNING:
                v=env.view
                key=(v,prior)
                if not physical:
                    if key in seen:break
                    seen.add(key)
                if key not in decisions:
                    if v not in observed:observed[v]=net.observed(v)
                    q,fields=net.physical_scores(v,observed[v],prior)
                    decisions[key]=(q[None,:]+params@fields.T).argmax(1)
                a=int(decisions[key][model])
                actions.append(a)
                prior=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                env.step(base.MOVES[a])
            outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
            wins[model]+=outcome=="SUCCESS"
            records.append(dict(map=row["name"],tier=row["tier"],model=model,outcome=outcome,steps=env.steps,
                refuels=env.refuels,repairs=env.repairs,prefix_steps=len(prefix),actions_sha256=sha256(actions).hexdigest()))
        if (k+1)%20==0:print(k+1,wins.tolist(),flush=True)
    return wins,records


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    rows=make_bank()
    def balanced(pack,n):
        return [r for t in ("H1","H2","H3","H4") for r in [c for c in pack if c["tier"]==t][:n]]
    fit,select=balanced(rows[:100],12),balanced(rows[100:],16)
    policy=ReturnPolicy()
    portfolio=np.asarray([[0]*9,[12,10,0,0,0,0,0,0,0],[16,1,8,0,2,0,6,0,0],
        [18,1,2,8,2,2,4,0,0],[18,0,6,4,0,0,4,6,0],[18,6,4,0,4,4,2,2,0],
        [18,2,5,1,4,0,0,0,8],[18,0,0,10,0,0,0,8,0]],float)
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    save(OUT/"protocol.json",dict(fit=[r["name"] for r in fit],select=[r["name"] for r in select],
         train_sha256=digest(OUT/"bank.jsonl"),base_sha256=WEIGHTS_HASH,portfolio=portfolio.tolist(),
         source_sha256={f:digest(HERE/f) for f in ("late_frontiers.cpp","late_frontiers.dll","return_policy.py","return_train.py")},
         expert_sha256=digest(EXP/"prev-two/adapter.py"),
         fit_target="terminal SUCCESS on suffix from first time both services were physically reached with super1500; explicit phase only in fit",
         inference="current observation plus previous real sensor pair; all candidate frontiers and roads recomputed from observed map",
         rule="3 best nonzero on common fit suffixes; compare these and their mean to zero on separate complete train starts; earliest on tie; frozen dev and ablation"))
    if not (OUT/"prefixes.json").exists():
        save(OUT/"prefixes.json",prefixes(fit))
    endings={r["map"]:r for r in json.loads((OUT/"prefixes.json").read_bytes())}
    def run(name,cases,params,suffixes=None,physical=False,ablate=False):
        p=OUT/name
        if p.exists():
            obj=json.loads(p.read_bytes())
            assert obj["coefficients"]==params.tolist()
            return np.asarray(obj["wins"])
        t=time.monotonic()
        wins,episodes=series(cases,params,policy,suffixes,physical,ablate)
        obj=dict(wins=wins.tolist(),coefficients=params.tolist(),episodes=episodes,
                 seconds=round(time.monotonic()-t,2),protocol_sha256=digest(OUT/"protocol.json"))
        if physical:
            obj.update(dev_sha256=digest(EXP/"data/dev.jsonl"),freeze_sha256=digest(OUT/"frozen.json"),ablation=ablate)
        save(p,obj)
        return wins
    scores=run("suffixes.json",fit,portfolio,endings)
    best=portfolio[np.argsort(-scores[1:],kind="stable")[:3]+1]
    options=np.vstack((portfolio[0],best,best.mean(0)))
    scores2=run("selection.json",select,options)
    i=int(scores2.argmax())
    save(OUT/"frozen.json",dict(coefficients=options[i].tolist(),base_sha256=WEIGHTS_HASH,
        protocol_sha256=digest(OUT/"protocol.json"),selection_sha256=digest(OUT/"selection.json"),
        fit=scores.tolist(),select=scores2.tolist()))
    print("fit",scores.tolist(),"select",scores2.tolist(),flush=True)
    dev=data.rows(EXP/"data/dev.jsonl")
    a=run("dev.physical.json",dev,options[i:i+1],physical=True)
    b=run("ablated.physical.json",dev,options[i:i+1],physical=True,ablate=True)
    print("with memory",a.tolist(),"same weights ablated",b.tolist(),flush=True)


if __name__=="__main__":main()
