"""Fit late action values on observed train handovers, then select on new full starts."""
from collections import Counter
from hashlib import sha256
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,moves,save
from service_window_train import LATE,LATE_HASH
from service_window_policy import coverage
from final_leg_policy import FinalLeg,load

OUT=HERE/"final-leg"


def dataset():
    a=data.rows(HERE/"return-search/bank.jsonl");b=data.rows(HERE/"handover-search/train.jsonl")
    candidates=data.read(HERE/"service-window/learned/fit.json")["episodes"]
    results={e["map"]:(e["first_late"] is not None,e["outcome"]) for e in candidates if e["model"]==0}
    handover=HERE/"handover-search"
    for name in data.read(handover/"protocol.json")["fit"][:36]:
        e=data.read(handover/"fit"/(name+".json"))["episodes"][0]
        results[name]=e["gate"],e["outcome"]
    indexed={r["name"]:r for r in a+b}
    train=[]
    for t in ("H1","H2","H3","H4"):
        group=[s for s in sorted(results) if indexed[s]["tier"]==t and results[s][0]]
        losses=[s for s in group if results[s][1]!="SUCCESS"]
        positives=[s for s in group if results[s][1]=="SUCCESS"]
        train.extend(indexed[s] for s in losses+positives[:max(2,len(losses)*2)])
    tiers=[[r for r in b[100:] if r["tier"]==t][:18] for t in ("H1","H2","H3","H4")]
    return train,[r for quad in zip(*tiers,strict=True) for r in quad]


def assess(tag,maps,options,net):
    dest=OUT/tag;dest.mkdir(exist_ok=True)
    wins=np.zeros(len(options),int);rows=[]
    for irow,r in enumerate(maps):
        file=dest/(r["name"]+".json")
        if file.exists():
            record=data.read(file);assert record["options"]==options.tolist()
            group=record["episodes"]
        else:
            env=Environment(data.level(r));early=bytearray();old=(None,None);visited=set()
            while env.status==Status.RUNNING and min(coverage(env.view.memory))<9:
                v=env.view;key=v,old
                if key in visited:break
                visited.add(key)
                a=int(net.early_scores(v,old).argmax());early.append(a)
                old=(v.fuel_distance,v.repair_distance);env.step(moves[a])
            reached=env.status==Status.RUNNING and min(coverage(env.view.memory))==9
            cache={};group=[]
            for j in range(len(options)):
                env.reset();old=(None,None)
                for a in early:
                    v=env.view;old=(v.fuel_distance,v.repair_distance)
                    env.step(moves[a])
                commands=bytearray(early);seen=set()
                while reached and env.status==Status.RUNNING:
                    v=env.view;key=v,old
                    if key in seen:break
                    seen.add(key)
                    if key not in cache:
                        q,f=net.late_parts(v,old)
                        cache[key]=(q[None,:]+options@f.T).argmax(1)
                    a=int(cache[key][j]);commands.append(a)
                    old=(v.fuel_distance,v.repair_distance);env.step(moves[a])
                group.append(dict(map=r["name"],tier=r["tier"],model=j,gate=reached,prefix=len(early),
                    outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle",steps=env.steps,
                    refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(commands).hexdigest()))
            save(file,dict(options=options.tolist(),episodes=group))
        wins+=np.array([e["outcome"]=="SUCCESS" for e in group]);rows.extend(group)
        if (irow+1)%10==0:print(tag,irow+1,wins.tolist(),flush=True)
    record=dict(options=options.tolist(),wins=wins.tolist(),episodes=rows)
    save(OUT/(tag+".json"),record);return wins


def physical(tag,rows,ablate):
    path=OUT/"model.pt";net=load(path.read_bytes(),ablate)
    dest=OUT/tag;dest.mkdir(exist_ok=True);episodes=[]
    for r in rows:
        p=dest/(r["name"]+".json")
        if p.exists():
            e=data.read(p);assert e["model_sha256"]==digest(path)
        else:
            env=Environment(data.level(r));commands=bytearray();cache={}
            while env.status==Status.RUNNING:
                m=net.choose(env.view,cache);commands.append(moves.index(m));env.step(m)
            e=dict(map=r["name"],tier=r["tier"],outcome=env.status.name,steps=env.steps,refuels=env.refuels,
                   repairs=env.repairs,actions_sha256=sha256(commands).hexdigest(),model_sha256=digest(path))
            save(p,e)
        episodes.append(e)
    save(OUT/(tag+".json"),dict(model_sha256=digest(path),dev_sha256=digest(EXP/"data/dev.jsonl"),ablated=ablate,
                              wins=[sum(e["outcome"]=="SUCCESS" for e in episodes)],episodes=episodes))


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    fit,select=dataset()
    options=np.array([[0]*9,[12,0,0,0,0,0,0,2,0],[12,0,0,0,0,0,0,5,2],
        [12,2,0,0,0,0,0,3,1],[12,1,1,2,0,0,0,2,1],[12,0,0,0,2,0,0,2,1],
        [12,0,0,2,0,1,2,3,1],[12,0,0,0,0,-4,0,2,1],[12,0,0,0,0,4,0,2,1],
        [12,3,0,3,0,0,0,0,0],[12,0,3,0,3,0,0,0,0]],float)
    assert digest(BASE)==BASE_HASH and digest(LATE)==LATE_HASH
    save(OUT/"protocol.json",dict(early=BASE_HASH,late=LATE_HASH,fit=[r["name"] for r in fit],select=[r["name"] for r in select],
        settings=options.tolist(),rule="Fit on all late failures of fixed gate in two predefined train subsets plus up to two same-tier successes per failure. Select by whole-trip SUCCESS on 72 held-back full train starts: top four and mean versus frozen baseline. First on ties. Ordinary early policy is fixed.",
        source={n:digest(HERE/n) for n in ("final_leg_policy.py","final_leg_train.py","proper_frontier.dll","service_window_policy.py")}))
    state=dict(format="full-window-final-leg-1",early=torch.load(BASE,weights_only=True,map_location="cpu"),late=torch.load(LATE,weights_only=True,map_location="cpu"),thresholds=torch.tensor([9,9,-101]))
    net=FinalLeg(dict(state,residual=torch.zeros(9,dtype=torch.float64)))
    a=assess("fit",fit,options,net)
    top=options[np.argsort(-a[1:],kind="stable")[:4]+1];finals=np.vstack((options[0],top,top.mean(0)))
    b=assess("select",select,finals,net);chosen=finals[int(b.argmax())]
    path=OUT/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp");torch.save(dict(state,residual=torch.tensor(chosen,dtype=torch.float64)),tmp);tmp.replace(path)
    save(OUT/"frozen.json",dict(weights=chosen.tolist(),model_sha256=digest(path),fit=a.tolist(),select=b.tolist(),
                               protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):
        physical(tag,data.rows(EXP/"data/dev.jsonl"),ablate)


if __name__=="__main__":main()
