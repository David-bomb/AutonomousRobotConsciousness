"""Terminal-reward search over frontier objectives. Writes a record after each train map."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,moves,save
from proper_policy import Policy

OUT=HERE/"proper-search"


def run(tag,rows,configs,policy,physical=False,ablate=False):
    folder=OUT/tag
    folder.mkdir(exist_ok=True)
    scores=np.zeros(len(configs),int)
    episodes=[]
    for i,r in enumerate(rows):
        file=folder/(r["name"]+".json")
        if file.exists():
            record=data.read(file)
            assert record["configs"]==configs and record["protocol_sha256"]==digest(OUT/"protocol.json")
            pair=record["episodes"]
        else:
            env=Environment(data.level(r))
            features={};actions={}
            pair=[]
            for j,setting in enumerate(configs):
                env.reset()
                old=(None,None);seen=set();commands=bytearray()
                while env.status==Status.RUNNING:
                    v=env.view;k=v,old
                    if not physical:
                        if k in seen:break
                        seen.add(k)
                    if k not in actions:
                        geometry=v.memory,v.position,v.fuel,v.health
                        if geometry not in features:features[geometry]=policy.masks(v)
                        actions[k]=policy.options(v,old,configs,features[geometry])
                    a=actions[k][j]
                    commands.append(a)
                    old=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                    env.step(moves[a])
                from hashlib import sha256
                pair.append(dict(map=r["name"],tier=r["tier"],policy=j,
                    outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle",
                    steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(commands).hexdigest()))
            save(file,dict(configs=configs,protocol_sha256=digest(OUT/"protocol.json"),episodes=pair))
        scores+=np.array([e["outcome"]=="SUCCESS" for e in pair])
        episodes.extend(pair)
        if (i+1)%10==0:print(tag,i+1,scores.tolist(),flush=True)
    summary=dict(configs=configs,wins=scores.tolist(),episodes=episodes)
    if physical:
        summary.update(weights_sha256=digest(OUT/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),ablate=ablate,
                       outcomes=dict(Counter(e["outcome"] for e in episodes)))
    save(OUT/(tag+".json"),summary)
    return scores


def main():
    torch.set_num_threads(1)
    assert digest(BASE)==BASE_HASH
    OUT.mkdir(exist_ok=True)
    corpus=HERE/"return-search/bank.jsonl"
    rows=data.rows(corpus)
    def balanced(group,start,n):
        tiers=[[r for r in group if r["tier"]==t][start:start+n] for t in ("H1","H2","H3","H4")]
        return [r for same in zip(*tiers,strict=True) for r in same]
    fit,select=balanced(rows[:100],4,8),balanced(rows[100:],7,13)
    configs=[[0,0,-101],[1,0,-101],[2,0,-101],[1,4,-101],[2,4,-101],[3,4,-101],
             [2,5,-101],[2,6,-101],[3,6,-101],[2,4,2],[3,6,2]]
    save(OUT/"protocol.json",dict(configs=configs,fit=[r["name"] for r in fit],select=[r["name"] for r in select],
         base_sha256=BASE_HASH,corpus_sha256=digest(corpus),
         method="Compare 11 discrete final action-score heads by terminal SUCCESS; four best nonzero heads against baseline on separate full train starts; earliest on ties. Freeze once before full dev/ablation.",
         observations="Native observation plus two previous actual F/R readings, reset on start. Only equal-cost shortest-route moves receive the learned frontier bonus; neural logits decide ties and unsupported positions.",
         source_sha256={n:digest(HERE/n) for n in ("renewable_features.cpp","proper_frontier.cpp","proper_frontier.dll","proper_policy.py","proper_train.py")}))
    controller=Policy(BASE.read_bytes())
    fit_scores=run("fit",fit,configs,controller)
    shortlist=[configs[0]]+[configs[i] for i in np.argsort(-fit_scores[1:],kind="stable")[:4]+1]
    selection=run("select",select,shortlist,controller)
    winner=shortlist[int(selection.argmax())]
    path=OUT/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp")
        torch.save(dict(format="proper-frontier-1",configuration=torch.tensor(winner,dtype=torch.int32),
                        base=torch.load(BASE,weights_only=True,map_location="cpu")),tmp)
        tmp.replace(path)
    save(OUT/"frozen.json",dict(setting=winner,train=fit_scores.tolist(),select=selection.tolist(),
                               model_sha256=digest(path),protocol_sha256=digest(OUT/"protocol.json")))
    print("frozen",winner,selection.tolist(),flush=True)
    for tag,ablate in (("dev",False),("ablated",True)):
        run(tag,data.rows(EXP/"data/dev.jsonl"),[winner],controller,physical=True,ablate=ablate)


if __name__=="__main__":main()
