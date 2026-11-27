"""Curriculum: reward a viable service prefix, select on complete terminal training trips."""
from hashlib import sha256
import numpy as np
import torch
from geodesic_probe import BASE,HASH,OUT,EXP,HERE,Environment,Status,data,moves
from geodesic_policy import Early
from renewable_train import digest,save
from scripts.audit_automaton import Reachability


def series(tag,rows,options,net,prefix=False,physical=False,ablate=False):
    folder=OUT/tag;folder.mkdir(exist_ok=True)
    scores=np.zeros(len(options),int)
    episodes=[]
    for i,row in enumerate(rows):
        artifact=folder/(row["name"]+".json")
        if artifact.exists():
            doc=data.read(artifact);assert doc["weights"]==options.tolist()
            group=doc["episodes"]
        else:
            env=Environment(data.level(row))
            possible=Reachability(env.level) if prefix else None
            cache={};group=[]
            for j in range(len(options)):
                env.reset();prior=(None,None);seen=set();actions=bytearray()
                while env.status==Status.RUNNING:
                    if prefix and env.refuels and env.repairs:break
                    v=env.view;k=v,prior
                    if not physical:
                        if k in seen:break
                        seen.add(k)
                    if k not in cache:
                        q,f=net.scores(v,prior)
                        cache[k]=(q[None,:]+options@f.T).argmax(1)
                    a=int(cache[k][j]);actions.append(a)
                    prior=(None,None) if ablate else (v.fuel_distance,v.repair_distance)
                    env.step(moves[a])
                reached=bool(env.refuels and env.repairs)
                good=env.status==Status.SUCCESS or bool(prefix and reached and env.status==Status.RUNNING and
                     possible.possible(env.position,env.fuel,env.health,env.limit-env.steps))
                outcome=env.status.name if env.status!=Status.RUNNING else ("both" if prefix and reached else "exact_cycle")
                group.append(dict(map=row["name"],tier=row["tier"],model=j,outcome=outcome,reward=int(good),
                                  steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest()))
            save(artifact,dict(weights=options.tolist(),episodes=group))
        scores+=np.array([r["reward"] for r in group]);episodes.extend(group)
        if (i+1)%20==0:print(tag,i+1,scores.tolist(),flush=True)
    result=dict(weights=options.tolist(),episodes=episodes,wins=scores.tolist())
    if physical:result.update(weights_sha256=digest(OUT/"model.pt"),dev_sha256=digest(EXP/"data/dev.jsonl"),ablated=ablate)
    save(OUT/(tag+".json"),result)
    return scores


def main():
    torch.set_num_threads(1)
    options=np.array([[0]*8,[2,1,0,0,1,0,10,0],[4,2,0,0,3,0,10,2],[8,4,1,0,4,1,12,4],
        [12,8,3,0,6,2,12,6],[6,5,2,1,5,2,12,6],[12,8,4,2,7,2,12,8],
        [2,1,0,0,10,4,12,5],[5,3,0,0,0,0,12,6],[12,8,6,4,12,6,16,10],
        [0,0,0,0,0,0,16,0],[0,0,0,0,10,0,16,0],[2,2,2,2,4,4,8,4]],float)
    rows=data.rows(EXP/"sequence/train.jsonl")
    def balanced(group,n):
        tiers=[[r for r in group if r["tier"]==t][:n] for t in ("H1","H2","H3","H4")]
        return [r for same in zip(*tiers,strict=True) for r in same]
    fit=balanced(rows[1000:1100],18);select=balanced(rows[600:700],17)
    sensor=OUT/"sensor-weights.json"
    save(OUT/"terminal-protocol.json",dict(fit=[r["name"] for r in fit],select=[r["name"] for r in select],
         base_sha256=HASH,sensor_sha256=digest(sensor),options=options.tolist(),
         rule="Fit reward: physical success or first joint service with a true viable continuation on train. Select: only terminal SUCCESS at end of full trips on separate train, best four nonzero prefix policies and their mean against zero; earliest on ties. No phase label in policy.",
         source={n:digest(HERE/n) for n in ("geodesic_evidence.dll","geodesic_probe.py","geodesic_train.py","geodesic_policy.py")}))
    weights=data.read(sensor)["weights"]
    assert digest(BASE)==HASH
    net=Early(BASE.read_bytes(),weights)
    reward=series("prefix",fit,options,net,prefix=True)
    top=options[np.argsort(-reward[1:],kind="stable")[:4]+1]
    finals=np.vstack([options[0],top,top.mean(0)])
    finish=series("terminal",select,finals,net)
    selected=finals[int(finish.argmax())]
    out=OUT/"model.pt"
    if not out.exists():
        tmp=out.with_suffix(".tmp")
        torch.save(dict(format="first-unknown-edge-1",base=torch.load(BASE,weights_only=True,map_location="cpu"),
                        sensor=torch.tensor(weights,dtype=torch.float64),mix=torch.tensor(selected,dtype=torch.float64)),tmp)
        tmp.replace(out)
    save(OUT/"frozen.json",dict(weights=selected.tolist(),prefix=reward.tolist(),terminal=finish.tolist(),
                               model_sha256=digest(out),protocol_sha256=digest(OUT/"terminal-protocol.json")))
    print("selected",selected.tolist(),"terminal",finish.tolist(),flush=True)
    for tag,ablate in (("dev",False),("ablated",True)):
        series(tag,data.rows(EXP/"data/dev.jsonl"),selected[None,:],net,physical=True,ablate=ablate)


if __name__=="__main__":main()
