"""Preregistered zero-shot composition and an independent terminal-reward gate search."""
from collections import Counter
from hashlib import sha256
import sys
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,moves,save
from service_window_policy import Mixture,coverage

OUT=HERE/"service-window"
LATE=EXP/"sequence/runs/joint400/rl.pt"
LATE_HASH="ebf419fe5ed73a9204b8cc976bedbc1ca53aeffe615605fb443f20fb4f852f56"


def freeze(out,base,cfg):
    p=out/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp")
        torch.save(dict(base,thresholds=torch.tensor(cfg,dtype=torch.int32)),tmp)
        tmp.replace(p)
    save(out/"frozen.json",dict(cfg=cfg,model_sha256=digest(p),protocol_sha256=digest(OUT/"protocol.json")))


def assess(out,rows,configs,net,physical=False,ablate=False):
    out.mkdir(exist_ok=True,parents=True)
    wins=np.zeros(len(configs),int);all_rows=[]
    for i,row in enumerate(rows):
        path=out/(row["name"]+".json")
        if path.exists():
            record=data.read(path)
            assert record["configs"]==configs
            pair=record["episodes"]
        else:
            env=Environment(data.level(row))
            pair=[];early={};late={}
            for j,cfg in enumerate(configs):
                env.reset();old=(None,None);seen=set();actions=bytearray()
                routed=0;first_route=None;both_step=None;premature=0;missed=0
                while env.status==Status.RUNNING:
                    v=env.view;k=v,old
                    if not physical:
                        if k in seen:break
                        seen.add(k)
                    use_late=net.route(v,cfg)
                    both=bool(env.refuels and env.repairs)
                    if both and both_step is None:both_step=env.steps
                    if use_late and first_route is None:first_route=env.steps
                    premature+=use_late and not both;missed+=both and not use_late
                    if use_late:
                        if v not in late:late[v]=int(net.late_scores(v).argmax())
                        a=late[v];routed+=1
                    else:
                        if k not in early:early[k]=int(net.early_scores(v,old).argmax())
                        a=early[k]
                    old=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                    actions.append(a);env.step(moves[a])
                pair.append(dict(map=row["name"],tier=row["tier"],model=j,outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle",
                    steps=env.steps,refuels=env.refuels,repairs=env.repairs,late_steps=routed,first_late=first_route,first_both=both_step,
                    before_both= int(premature),after_both_early=int(missed),actions_sha256=sha256(actions).hexdigest()))
            save(path,dict(configs=configs,episodes=pair))
        wins+=np.array([v["outcome"]=="SUCCESS" for v in pair])
        all_rows.extend(pair)
        if (i+1)%20==0:print(out.name,i+1,wins.tolist(),flush=True)
    summary=dict(configs=configs,wins=wins.tolist(),episodes=all_rows)
    if physical:
        summary.update(weights_sha256=digest(out.parent/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),
           ablate=ablate,outcomes=dict(Counter(r["outcome"] for r in all_rows)),
           tiers={t:sum(r["outcome"]=="SUCCESS" for r in all_rows if r["tier"]==t) for t in ("H1","H2","H3","H4")},
           both=sum(bool(r["refuels"] and r["repairs"]) for r in all_rows))
    save(out.with_suffix(".json"),summary)
    return wins


def main(which):
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    assert digest(BASE)==BASE_HASH and digest(LATE)==LATE_HASH
    bank=data.rows(HERE/"return-search/bank.jsonl")
    def balanced(rows,n):
        tiers=[[r for r in rows if r["tier"]==t][:n] for t in ("H1","H2","H3","H4")]
        return [r for col in zip(*tiers,strict=True) for r in col]
    fit,select=balanced(bank[:100],12),balanced(bank[100:],18)
    original=[9,9,-101]
    configs=[original,[8,9,-101],[7,9,-101],[8,8,-101],[6,8,-101],[9,9,0],[9,9,2],[8,9,2],
             [7,9,2],[6,8,2],[9,9,5],[0,0,-101],[10,10,-101]]
    save(OUT/"protocol.json",dict(base_sha256=BASE_HASH,late_sha256=LATE_HASH,
        zero_shot=original,configs=configs,fit=[r["name"] for r in fit],select=[r["name"] for r in select],
        method="Zero-shot fixed before scores: late expert if one fuel and one repair station each have a fully observed 3x3 window. Separate training population: terminal success on train, then top four plus fixed gate and two parent policies on separate full train starts, first on ties. Two previous real readings only.",
        sources={n:digest(HERE/n) for n in ("service_window_policy.py","service_window_train.py")},
        component_sources={str(p.relative_to(EXP)):digest(p) for p in
           (EXP/"prev-two/adapter.py",EXP/"sequence/belief/joint_model.py",EXP/"sequence/belief/itineraries.dll")}))
    base=dict(format="service-window-mixture-1",early=torch.load(BASE,weights_only=True,map_location="cpu"),
              late=torch.load(LATE,weights_only=True,map_location="cpu"))
    net=Mixture(dict(base,thresholds=torch.tensor(original)))
    dest=OUT/which
    dest.mkdir(exist_ok=True)
    if which=="fixed":
        selected=original
    elif which=="learned":
        first=assess(dest/"fit",fit,configs,net)
        best=np.argsort(-first[:-2],kind="stable")[:4]
        shortlist=[original]+[configs[i] for i in best if configs[i]!=original]+configs[-2:]
        second=assess(dest/"select",select,shortlist,net)
        selected=shortlist[int(second.argmax())]
    else:raise ValueError(which)
    freeze(dest,base,selected)
    for tag,ablate in (("dev",False),("ablated",True)):
        assess(dest/tag,data.rows(EXP/"data/dev.jsonl"),[selected],net,True,ablate)


if __name__=="__main__":main(sys.argv[1])
