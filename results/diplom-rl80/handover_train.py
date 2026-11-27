"""Assign whole-trip terminal credit to the early head with a fixed, observation-only handover."""
from hashlib import sha256
import json
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,moves,save
from service_window_train import LATE,LATE_HASH
from service_window_policy import coverage
from handover_policy import Handover,load
from maze.datasets import geometry

OUT=HERE/"handover-search"


def new_bank():
    p=OUT/"train.jsonl"
    if p.exists():return data.rows(p)
    used=data.excluded()
    for path in (EXP/"data/train.jsonl",EXP/"data/dev.jsonl",EXP/"data/confirmation.jsonl",
                 EXP/"sequence/train.jsonl",EXP/"sequence/probe.jsonl",EXP/"sequence/spatial/extra.jsonl",
                 HERE/"return-search/bank.jsonl"):
        if path.exists():used.update(r["geometry"] for r in data.rows(path))
    out=[]
    for seed in (9165410,9165411):
        for r in data.generate(("handover-train",seed)):
            h=geometry(r["grid"])
            assert h==r["geometry"] and h not in used
            used.add(h);out.append(r)
    tmp=p.with_suffix(".tmp");tmp.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in out),encoding="utf8")
    tmp.replace(p)
    return out


def complete(env,policy,actions,values):
    trail=[];seen=set()
    while env.status==Status.RUNNING:
        v=env.view
        assert min(coverage(v.memory))==9
        key=(v,env.limit-env.steps)
        if key in values:
            outcome,n,f,r=values[key];break
        if v in seen:
            outcome,n,f,r="exact_cycle",0,0,0;break
        seen.add(v)
        trail.append((key,env.refuels,env.repairs))
        if v not in actions:actions[v]=int(policy.late_scores(v).argmax())
        env.step(moves[actions[v]])
    else:outcome,n,f,r=env.status.name,0,0,0
    end_f,end_r=env.refuels,env.repairs
    for k,old_f,old_r in reversed(trail):
        n+=1;f+=end_f-old_f;r+=end_r-old_r
        end_f,end_r=old_f,old_r
        values[k]=(outcome,n,f,r)
    return outcome,n,f,r


def learn(stage,rows,options,net):
    dest=OUT/stage;dest.mkdir(exist_ok=True)
    wins=np.zeros(len(options),int);reached=np.zeros(len(options),int);all_rows=[]
    for i,r in enumerate(rows):
        p=dest/(r["name"]+".json")
        if p.exists():
            blob=data.read(p);assert blob["options"]==options.tolist()
            runs=blob["episodes"]
        else:
            env=Environment(data.level(r))
            early={};late={};values={};runs=[]
            for j in range(len(options)):
                env.reset();old=(None,None);seen=set();started=bytearray()
                while env.status==Status.RUNNING and min(coverage(env.view.memory))<9:
                    v=env.view;key=v,old
                    if key in seen:break
                    seen.add(key)
                    if key not in early:
                        q,parts=net.early_parts(v,old)
                        early[key]=(q[None,:]+options@parts.T).argmax(1)
                    a=int(early[key][j]);started.append(a)
                    old=(v.fuel_distance,v.repair_distance);env.step(moves[a])
                handoff=env.status==Status.RUNNING and min(coverage(env.view.memory))==9
                pf,pr=env.refuels,env.repairs
                if handoff:outcome,n,f,rep=complete(env,net,late,values)
                else:outcome,n,f,rep=(env.status.name if env.status!=Status.RUNNING else "exact_cycle"),0,0,0
                runs.append(dict(map=r["name"],tier=r["tier"],model=j,outcome=outcome,gate=handoff,
                                 prefix=len(started),prefix_sha256=sha256(started).hexdigest(),late_steps=n,
                                 refuels=pf+f,repairs=pr+rep))
            save(p,dict(options=options.tolist(),episodes=runs,distinct_early=len(early),distinct_late=len(late),
                        protocol_sha256=digest(OUT/"protocol.json")))
        wins+=np.array([e["outcome"]=="SUCCESS" for e in runs])
        reached+=np.array([e["gate"] for e in runs])
        all_rows.extend(runs)
        if (i+1)%12==0:print(stage,i+1,wins.tolist(),flush=True)
    save(OUT/(stage+".json"),dict(options=options.tolist(),wins=wins.tolist(),gates=reached.tolist(),episodes=all_rows))
    return wins


def physical(tag,rows,ablate):
    out=OUT/tag;out.mkdir(exist_ok=True)
    model=load((OUT/"model.pt").read_bytes(),ablate)
    episodes=[]
    for i,r in enumerate(rows):
        p=out/(r["name"]+".json")
        if p.exists():
            e=data.read(p);assert e["model_sha256"]==digest(OUT/"model.pt")
        else:
            env=Environment(data.level(r));cache={};actions=bytearray()
            while env.status==Status.RUNNING:
                m=model.choose(env.view,cache)
                actions.append(moves.index(m));env.step(m)
            e=dict(map=r["name"],tier=r["tier"],geometry=r["geometry"],outcome=env.status.name,
                steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest(),
                model_sha256=digest(OUT/"model.pt"))
            save(p,e)
        episodes.append(e)
        if (i+1)%25==0:print(tag,i+1,sum(e["outcome"]=="SUCCESS" for e in episodes),flush=True)
    save(OUT/(tag+".json"),dict(model_sha256=digest(OUT/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),
             wins=[sum(e["outcome"]=="SUCCESS" for e in episodes)],ablated=ablate,episodes=episodes))


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert digest(BASE)==BASE_HASH and digest(LATE)==LATE_HASH
    bank=new_bank()
    def balanced(rows,n):
        tiers=[[r for r in rows if r["tier"]==t][:n] for t in ("H1","H2","H3","H4")]
        return [r for four in zip(*tiers,strict=True) for r in four]
    fit,select=balanced(bank[:100],15),balanced(bank[100:],22)
    generator=np.random.default_rng(271190)
    settings=[[0]*10,[0,0,0,0,6,5,2,5,12,0],[0,0,0,0,12,8,3,6,12,0],
              [0,3,2,1,4,2,0,2,12,0],[0,6,0,0,0,0,0,0,0,0],
              [1,-2,5,3,0,0,0,3,10,1],[-1,5,2,1,3,2,1,4,8,2]]
    for _ in range(15):
        settings.append(generator.uniform([-2,-3,-2,-2,0,0,0,0,0,-1],
                                          [3,8,6,4,12,8,5,10,16,3]).round(2).tolist())
    options=np.array(settings,float)
    ss=HERE/"geodesic-search/sensor-weights.json"
    save(OUT/"protocol.json",dict(fit=[r["name"] for r in fit],select=[r["name"] for r in select],options=settings,
        input_hash={str(p.name):digest(p) for p in (BASE,LATE,ss,OUT/"train.jsonl",HERE/"handover_policy.py",
                                  HERE/"handover_train.py",HERE/"service_window_policy.py",HERE/"geodesic_evidence.dll")},
        rule="Constant 9/9 observed service windows. Fit 22 action-score heads by physical whole-trip SUCCESS, then select four best nonzero, their average and zero on 88 separate train maps. First on ties. No hidden phase or physical counters at evaluation.",
        caching="Only deterministic training suffixes memoized for the tuple (identical physical Observation, identical remaining step limit) within one map. Exact cycles imply zero reward. Dev runs each physical step through choose."))
    base=dict(format="trained-handover-1",early=torch.load(BASE,weights_only=True,map_location="cpu"),
              late=torch.load(LATE,weights_only=True,map_location="cpu"),thresholds=torch.tensor([9,9,-101]),
              sensor=torch.tensor(data.read(ss)["weights"],dtype=torch.float64))
    net=Handover(dict(base,residual=torch.zeros(10,dtype=torch.float64)))
    first=learn("fit",fit,options,net)
    leaders=options[np.argsort(-first[1:],kind="stable")[:4]+1]
    final=np.vstack((options[0],leaders,leaders.mean(0)))
    result=learn("select",select,final,net)
    best=final[int(result.argmax())]
    p=OUT/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp");torch.save(dict(base,residual=torch.tensor(best,dtype=torch.float64)),tmp);tmp.replace(p)
    save(OUT/"frozen.json",dict(weights=best.tolist(),model_sha256=digest(p),protocol_sha256=digest(OUT/"protocol.json"),
                               fit=first.tolist(),select=result.tolist()))
    dev=data.rows(EXP/"data/dev.jsonl")
    physical("dev",dev,False);physical("ablated",dev,True)


if __name__=="__main__":main()
