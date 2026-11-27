"""Fit a sensor-direction likelihood on train and check it on held-back on-policy traversals."""
import ctypes
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
import torch
from renewable_policy import targets

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.environment import Environment,Status
from maze.grid import destination,wave,locations
from maze.interface import Move

data=import_module("results.E-0016.data")
prev=import_module("results.E-0016.prev-two.adapter")
EXP=PROJECT/"results/E-0016"
OUT=HERE/"geodesic-search"
BASE=EXP/"prev-two/weights/wave1.pt"
HASH="369c0d5616a07a19f6fa992d53bdaa5a3004c5c54344c427f2345272b2f56432"
moves=tuple(Move)


class Evidence:
    def __init__(self):
        self.lib=ctypes.CDLL(str(HERE/"geodesic_evidence.dll"))
        ci=ctypes.c_int
        ints=np.ctypeslib.ndpointer(np.int32)
        self.lib.evidence.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,np.ctypeslib.ndpointer(np.float64)]
        self.lib.evidence.restype=None

    def features(self,v,past):
        x,y=v.position;w=len(v.memory[0])
        old=y*w+x
        if v.moved:
            dx,dy=v.last_move.value
            old-=dy*w+dx
        sites=targets(v.memory,(x+v.dx,y+v.dy))
        now=np.array([v.fuel_distance,v.repair_distance],dtype=np.int32)
        before=np.array([-1 if d is None else d for d in past],dtype=np.int32)
        m=np.empty((8,2,9))
        self.lib.evidence("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,old,sites,now,before,m)
        for k in range(2,8):
            total=m[:,:,k].sum(0)
            m[:,:,k]/=np.maximum(total,1.e-7)
        return m


def traces(rows,source,commands,policy,est):
    fields=[];stats=[];targets_all=[]
    for i,r in enumerate(rows):
        env=Environment(data.level(r))
        truth=[{p:d for p,d,_ in wave(env.level.grid,locations(env.level.grid,c))} for c in "FR"]
        past=(None,None)
        script=commands.get(r["name"])
        while env.status==Status.RUNNING and env.steps<50 and not (env.refuels and env.repairs):
            v=env.view
            if script is not None and env.steps>=len(script):break
            vals=policy.net.features(v,past)
            legal=vals[-1]
            tensor=prev.tensor_batch([vals])
            with torch.inference_mode():
                q,signs,_=policy.net.heads(*tensor)
                prob=signs[0].softmax(-1)[:,:,0].numpy()
                if v.health<v.fuel:prob=prob[:,::-1]
                q=q[0].numpy()
            chosen=int(q.argmax()) if script is None else int(script[env.steps])
            ds=[v.fuel_distance,v.repair_distance]
            stocks=[v.fuel,v.health]
            kind=min((k for k in (0,1) if ds[k]>0),key=lambda k:stocks[k]-ds[k],default=None)
            if legal.sum()>1 and kind is not None and stocks[kind]-ds[kind]<=12:
                infos=est.features(v,past)[:,kind]
                real=np.zeros(8,bool)
                for a,m in enumerate(moves):
                    if legal[a]:
                        p=destination(v.memory,v.position,m)
                        real[a]=truth[kind][p]<ds[kind]
                assert real.any() and (infos[real,1]==1).all(),(r["name"],env.steps,real.tolist(),infos[:,1].tolist())
                assert not ((infos[:,0]>0)&~real).any(),(r["name"],env.steps)
                logits=(q-q[legal].max())/max(.5,float(q[legal].std()))
                z=np.column_stack((infos,prob[:,kind],np.clip(logits,-3,0)))
                fields.append(z.astype(np.float32))
                targets_all.append(real)
                stats.append(dict(map=r["name"],tier=r["tier"],step=env.steps,kind=kind,slack=stocks[kind]-ds[kind],
                                  source=source,model=int(q.argmax()),sensor=int(np.where(legal,prob[:,kind],-1).argmax()),
                                  legal=legal.nonzero()[0].tolist()))
            past=tuple(ds)
            env.step(moves[chosen])
        if (i+1)%50==0:print(source,i+1,"maps",len(stats),"samples",flush=True)
    return np.asarray(fields),np.asarray(targets_all),stats


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    est=Evidence()
    policy=prev.Controller(BASE,HASH)
    rows=data.rows(EXP/"sequence/train.jsonl")
    commands={r["map"]:r["actions"] for r in data.read(EXP/"sequence/clean-400.json")["episodes"]}
    packs=[("train-demo",rows[:200],commands),("train-rollout",rows[500:600],{}),
           ("probe-rollout",rows[1500:1600],{})]
    source_sha={n:sha256((HERE/n).read_bytes()).hexdigest() for n in
                ("geodesic_evidence.cpp","geodesic_evidence.dll","geodesic_probe.py")}
    plan=dict(train={tag:[r["name"] for r in sub] for tag,sub,_ in packs},base=HASH,source=source_sha,
              task="predict a true distance-decreasing action to most constrained positive service on a legal fork, before both services, first 50 moves, slack<=12",
              method="shared linear softmax on 11 observed features, uniform positives; equal total weight per map; held-back probe never fits coefficients")
    pp=OUT/"sensor-protocol.json"
    if pp.exists():assert data.read(pp)==plan
    else:data.save(pp,plan)
    arrays=[]
    for tag,sub,scripts in packs:
        file=OUT/(tag+".npz")
        detail=file.with_suffix(".json")
        if file.exists() and detail.exists():
            arr=np.load(file)
            arrays.append((arr["x"],arr["y"],data.read(detail)))
            continue
        x,y,s=traces(sub,tag,scripts,policy,est)
        np.savez_compressed(file,x=x,y=y)
        data.save(detail,s)
        arrays.append((x,y,s))
    x=torch.from_numpy(np.concatenate([a[0] for a in arrays[:2]]))
    y=torch.from_numpy(np.concatenate([a[1] for a in arrays[:2]]))
    from collections import Counter
    names=[s["map"] for a in arrays[:2] for s in a[2]]
    counts=Counter(names)
    weight=torch.tensor([1/counts[n] for n in names])
    coef=torch.zeros(x.shape[2],requires_grad=True)
    optimizer=torch.optim.Adam([coef],lr=.06)
    for _ in range(350):
        pred=(x@coef).masked_fill(x[:,:,1]==0,-1000)
        loss=(-(pred.log_softmax(1)*y).sum(1)/y.sum(1)*weight).sum()/weight.sum()+.0008*coef.square().sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
    frozen=coef.detach().numpy()
    model=OUT/"sensor-weights.json"
    data.save(model,dict(weights=frozen.tolist(),train_sha256=sha256(pp.read_bytes()).hexdigest()))
    xx,yy,st=arrays[-1]
    prediction=np.where(xx[:,:,1]>0,xx@frozen,-1000).argmax(1)
    for i,s in enumerate(st):
        s.update(pred=int(prediction[i]),actual=yy[i].nonzero()[0].tolist(),eligible=(xx[i,:,1]>0).nonzero()[0].tolist(),
                 certain=(xx[i,:,0]>0).nonzero()[0].tolist())
    def counts_for(group):
        return dict(n=len(group),maps=len({r["map"] for r in group}),
                    **{k:sum(r[k] in r["actual"] for r in group) for k in ("pred","model","sensor")},
                    single=sum(len(r["eligible"])==1 for r in group),certified=sum(bool(r["certain"]) for r in group))
    first={}
    for s in st:
        if s["slack"]<=2:first.setdefault(s["map"],s)
    groups={"all":counts_for(st),"slack<=2":counts_for([s for s in st if s["slack"]<=2]),
            "first_slack<=2_per_map":counts_for(list(first.values()))}
    data.save(OUT/"sensor-probe.json",dict(groups=groups,observations=st))
    print(groups,flush=True)


if __name__=="__main__":main()
