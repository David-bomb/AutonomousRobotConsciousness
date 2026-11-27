"""Policy search on terminal SUCCESS with common training maps and one preselected dev check."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import time

import numpy as np
import torch
from prefix_controller import HERE, PROJECT, PrefixPolicy, base
from maze.environment import Environment, Status

data=import_module("results.E-0016.data")
OUT=HERE/"prefix-search"
EXP=PROJECT/"results/E-0016"
BANK=HERE/"gradient-prefix.bin"
BASE=EXP/"runs/recovery800/rl.pt"
SEED=274102


def save(name, value):
    path=OUT/name
    body=(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=body:
        raise ValueError(f"Different existing record {name}")
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)


def episodes(rows, parameters, controller, physical):
    scores=np.zeros(len(parameters),int)
    records=[]
    for r in rows:
        lvl=data.level(r)
        cache={}
        env=Environment(lvl)
        for i,coeff in enumerate(parameters):
            env.reset()
            seen=set()
            same=False
            while env.status==Status.RUNNING:
                view=env.view
                if not physical:
                    if view in seen:
                        same=True
                        break
                    seen.add(view)
                if view not in cache:
                    base_scores,fields=controller.features(view)
                    cache[view]=(base_scores[None,:]+parameters@fields.T).argmax(1)
                move=base.MOVES[int(cache[view][i])]
                env.step(move)
            won=env.status==Status.SUCCESS
            scores[i]+=won
            records.append(dict(model=i,map=r["name"],tier=r["tier"],outcome="cycle" if same else env.status.name,
                                steps=env.steps,refuels=env.refuels,repairs=env.repairs))
    return scores,records


def sample(rows, rng, n):
    return [rows[int(i)] for tier in ("H1","H2","H3","H4")
            for i in rng.choice([j for j,r in enumerate(rows) if r["tier"]==tier],n,replace=False)]


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    train=data.rows(EXP/"data/train.jsonl")[:600]
    train_fit=[r for i,r in enumerate(train) if i%5!=0]
    train_select=[r for i,r in enumerate(train) if i%5==0]
    controller=PrefixPolicy(BASE,BANK)
    center=np.array([8.,4.,1.,0.,2.,3.,4.,0.,6.,1.,1.,0.])
    spread=np.array([4.,3.,2.,1.5,2.,2.,2.,2.,3.,2.,3.,2.])
    retained=[np.zeros_like(center),center.copy()]
    sig=lambda p:sha256(p.read_bytes()).hexdigest()
    save("protocol.json",dict(seed=SEED,base=sig(BASE),bank=sig(BANK),
         sources={s:sig(HERE/s) for s in ("gradient_prefix.cpp","gradient_prefix.dll","prefix_controller.py","prefix_search.py")},
         fit="E-0016 train[0:600], indices not divisible by 5",selection="120 remaining maps of train[0:600]",
         reward="1 for physical SUCCESS, 0 for other outcomes; deterministic repeated view is a loss during fit only",
         rounds=4,population=12,maps_per_round=32,ranking="mean of top four terminal counts, earlier on ties",
         final_selection="maximum train-select successes, retain earliest on ties; dev once after registration"))
    for g in range(4):
        record=OUT/f"round-{g+1}.json"
        noise=rng.standard_normal((5,len(center)))
        batch=sample(train_fit,rng,8)
        variants=np.vstack((center,retained[0],center+spread*noise,center-spread*noise))
        if record.exists():
            old=json.loads(record.read_bytes())
            assert old["coefficients"]==variants.tolist() and old["maps"]==[r["name"] for r in batch]
            scores=np.asarray(old["wins"])
        else:
            start=time.monotonic()
            scores,records=episodes(batch,variants,controller,False)
            save(record.name,dict(maps=[r["name"] for r in batch],coefficients=variants.tolist(),wins=scores.tolist(),
                                  episodes=records,elapsed_seconds=round(time.monotonic()-start,2)))
        rank=np.argsort(-scores,kind="stable")
        retained.append(variants[rank[0]].copy())
        center=variants[rank[:4]].mean(0)
        spread=np.maximum(.8,spread*.80)
        print("round",g+1,scores.tolist(),flush=True)
    retained.append(center)
    candidates=np.stack(retained)
    step=OUT/"selection.json"
    if step.exists():
        old=json.loads(step.read_bytes())
        assert old["coefficients"]==candidates.tolist()
        scores=np.asarray(old["wins"])
    else:
        scores,records=episodes(train_select,candidates,controller,False)
        save(step.name,dict(coefficients=candidates.tolist(),wins=scores.tolist(),episodes=records))
    selected=int(scores.argmax())
    save("frozen.json",dict(coefficients=retained[selected].tolist(),candidate=selected,training_successes=int(scores[selected]),
         checkpoint_sha256=sig(BASE),bank_sha256=sig(BANK),selection_sha256=sig(step),protocol_sha256=sig(OUT/"protocol.json")))
    print("train selection",scores.tolist(),"selected",selected,flush=True)
    rows=data.rows(EXP/"data/dev.jsonl")
    score,records=episodes(rows,candidates[selected:selected+1],controller,True)
    groups=Counter(r["tier"] for r in records if r["outcome"]=="SUCCESS")
    save("dev.physical.json",dict(frozen_sha256=sig(OUT/"frozen.json"),dev_sha256=sig(EXP/"data/dev.jsonl"),
                                  wins=int(score[0]),by_tier=dict(groups),episodes=records))
    print("physical dev",int(score[0]),dict(groups),flush=True)


if __name__=="__main__":
    main()
