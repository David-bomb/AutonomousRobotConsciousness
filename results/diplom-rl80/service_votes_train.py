"""Terminal-reward fit of ordinary numerical road votes before the fixed handover."""
from collections import Counter
from hashlib import sha256
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,moves,save
from service_window_train import LATE,LATE_HASH
from service_window_policy import coverage
from service_votes_policy import VoteMixture,load
from handover_train import complete

OUT=HERE/"service-votes"


def evaluate(tag,rows,options,net,physical=False,ablate=False):
    dest=OUT/tag;dest.mkdir(exist_ok=True)
    wins=np.zeros(len(options),int);all_rows=[]
    for i,r in enumerate(rows):
        p=dest/(r["name"]+".json")
        if p.exists():
            existing=data.read(p);assert existing["settings"]==options.tolist()
            batch=existing["episodes"]
        else:
            env=Environment(data.level(r));cache={};late={};values={};batch=[]
            for j in range(len(options)):
                env.reset();prior=(None,None);seen=set();command=bytearray()
                while env.status==Status.RUNNING:
                    v=env.view;key=v,prior
                    if not physical:
                        if key in seen or min(coverage(v.memory))==9:break
                        seen.add(key)
                    if min(coverage(v.memory))==9:
                        if v not in late:late[v]=int(net.late_scores(v).argmax())
                        a=late[v]
                    else:
                        if key not in cache:
                            q,f=net.early_parts(v,prior)
                            cache[key]=(q[None,:]+options@f.T).argmax(1)
                        a=int(cache[key][j])
                    prior=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                    command.append(a);env.step(moves[a])
                steps,fuels,repairs=env.steps,env.refuels,env.repairs
                if not physical and env.status==Status.RUNNING and min(coverage(env.view.memory))==9:
                    outcome,n,f,rep=complete(env,net,late,values)
                    steps+=n;fuels+=f;repairs+=rep
                else:outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
                row=dict(map=r["name"],tier=r["tier"],model=j,outcome=outcome,steps=steps,refuels=fuels,repairs=repairs)
                row["actions_sha256" if physical else "prefix_sha256"]=sha256(command).hexdigest()
                batch.append(row)
            save(p,dict(settings=options.tolist(),episodes=batch))
        wins+=np.array([z["outcome"]=="SUCCESS" for z in batch]);all_rows.extend(batch)
        if (i+1)%12==0:print(tag,i+1,wins.tolist(),flush=True)
    result=dict(settings=options.tolist(),wins=wins.tolist(),episodes=all_rows)
    if physical:
        result.update(model_sha256=digest(OUT/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),
                      outcomes=dict(Counter(z["outcome"] for z in all_rows)),ablated=ablate)
    save(OUT/(tag+".json"),result)
    return wins


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert digest(BASE)==BASE_HASH and digest(LATE)==LATE_HASH
    # Separate from the two hundred training maps used for diagnosis.
    rows=data.rows(HERE/"return-search/bank.jsonl")
    def balanced(group,start,n):
        tiers=[[r for r in group if r["tier"]==t][start:start+n] for t in ("H1","H2","H3","H4")]
        return [r for quad in zip(*tiers,strict=True) for r in quad]
    fit,select=balanced(rows[:100],10,13),balanced(rows[100:],3,18)
    options=np.array([[0]*10,[2,1,0,2,4,0,0,0,0,0],[5,2,0,4,6,0,0,0,0,0],
        [10,5,1,6,12,0,0,0,0,0],[16,10,4,12,12,0,0,0,0,0],[5,2,1,2,6,4,0,0,0,0],
        [5,2,1,2,6,0,4,0,0,0],[4,2,0,2,8,0,0,5,0,0],[4,1,0,5,10,0,0,0,5,0],
        [4,1,0,5,10,0,0,0,0,5],[0,0,0,10,12,0,0,0,0,0],[7,3,1,5,5,2,2,1,1,1],
        [3,1,0,6,10,5,0,3,2,1],[3,1,0,6,10,0,5,3,2,1]],float)
    source=EXP.parents[1]/"maze"
    save(OUT/"protocol.json",dict(fit=[r["name"] for r in fit],select=[r["name"] for r in select],settings=options.tolist(),
        early_sha256=BASE_HASH,late_sha256=LATE_HASH,training="Whole physical terminal SUCCESS, first 52 train maps, best four nonzero and their mean against baseline on separate 72 full train starts. Fixed 9/9 handover. First on ties.",
        features="Numerical first-move probabilities from the historical observed-road model plus known-path certificates, combined by trainable linear action scores. Its choose method is never called. Its original empirical design saw historical controls; the new fitting and selection use fresh training maps.",
        sources={f.name:digest(f) for f in [HERE/"service_votes_policy.py",HERE/"service_votes_train.py",HERE/"service_window_policy.py",
                                           source/"mandatory_supply.py",source/"street_trunk.py",source/"proven_approach.py"]}))
    core=dict(format="service-vote-mixture-1",early=torch.load(BASE,weights_only=True,map_location="cpu"),
              late=torch.load(LATE,weights_only=True,map_location="cpu"),thresholds=torch.tensor([9,9,-101]))
    net=VoteMixture(dict(core,residual=torch.zeros(10,dtype=torch.float64)))
    first=evaluate("fit",fit,options,net)
    leaders=options[np.argsort(-first[1:],kind="stable")[:4]+1]
    pool=np.vstack((options[0],leaders,leaders.mean(0)))
    second=evaluate("select",select,pool,net)
    selected=pool[int(second.argmax())]
    out=OUT/"model.pt"
    if not out.exists():
        tmp=out.with_suffix(".tmp");torch.save(dict(core,residual=torch.tensor(selected,dtype=torch.float64)),tmp);tmp.replace(out)
    save(OUT/"frozen.json",dict(weights=selected.tolist(),model_sha256=digest(out),protocol_sha256=digest(OUT/"protocol.json"),
                               fit=first.tolist(),select=second.tolist()))
    full=load(out.read_bytes())
    dev=data.rows(EXP/"data/dev.jsonl")
    evaluate("dev",dev,selected[None,:],full,True,False);evaluate("ablated",dev,selected[None,:],full,True,True)


if __name__=="__main__":main()
