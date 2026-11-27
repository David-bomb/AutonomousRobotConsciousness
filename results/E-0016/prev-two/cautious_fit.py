"""Подгонка нелинейной поправки на сильных ошибках и сопоставимых верных решениях wave1."""
import os
for k in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import sys
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment
from . import early_data
from .adapter import MOVES
from .cautious_bank import OUT,bank
from .cautious_policy import Correction,base_parts,design,evidence
from .common import HERE,SEED,digest,level,read,write
from .early_bank import LATE,LATE_SHA

FIELDS=("x","legal","own","viable","sign","best","tier","seed")


def contrast(v,previous,e,base,event):
    k=event["urgent"];legal=base[1];sign=np.array(event["signs"])[:,k]
    d=(v.fuel_distance,v.repair_distance)
    urgent=list(previous);urgent[1-k]=None
    flipped=tuple(None if p is None else max(0,2*now-p) for p,now in zip(previous,d))
    variants={"correct":e,"urgent_only":evidence(v,urgent),"missing":evidence(v,(None,None)),"reversed_time":evidence(v,flipped)}
    out={}
    for name,data in variants.items():
        allowed=((data[:,k,0]>.5)|(data[:,k,9]>0))&legal
        score=data[:,k,0]*100+data[:,k,11]/max(data[:,k,11].sum(),1e-6)
        score=np.where(allowed,score,-10)
        # Знаменатель включает все фактически сокращающие направления.
        m=Counter(states=1,true_down=int(((sign==0)&legal).sum()),retained=int(((sign==0)&allowed).sum()),
                  candidates=int(allowed.sum()),top_down=int(sign[score.argmax()]==0),no_candidates=int(not allowed.any()))
        out[name]=m
    return out


def encode(row):
    trip=read(OUT/"prefix"/(row["name"]+".json"));events={e["step"]:e for e in trip["events"]}
    env=Environment(level(row));previous=(None,None);samples=[];counts={}
    for step,c in enumerate(trip["actions"]):
        v=env.view
        if step in events:
            event=events[step];assert event["previous"]==list(previous)
            base=base_parts(early_data.NET,v,previous)
            e=evidence(v,previous);x,gate=design(v,previous,e,base)
            own=event["own"];legal=base[1];k=event["urgent"]
            assert int(base[0].argmax())==own
            if gate:
                true=np.asarray(event["signs"],np.int64)[:,[k,1-k]]
                viable=np.array(event["viable"])
                score=np.where(viable,true[:,0],6)
                best=viable&(true[:,0]==score.min())
                samples.append((x.astype("f2"),legal,own,viable,true,best,int(row["tier"][1]),int(row["name"].split("-")[-3])))
            for mode,cnt in contrast(v,previous,e,base,event).items():
                counts.setdefault(mode,Counter()).update(cnt)
        previous=v.fuel_distance,v.repair_distance
        env.step(MOVES[int(c)])
    dest=OUT/"encoded"/(row["name"]+".npz");dest.parent.mkdir(exist_ok=True)
    if samples:np.savez_compressed(dest,**dict(zip(FIELDS,map(np.asarray,zip(*samples)),strict=True)))
    return row["name"],dict(n=len(samples),sha256=digest(dest) if samples else None,contrasts=counts)


def collect():
    with ProcessPoolExecutor(2,initializer=early_data.init,initargs=("wave1",)) as pool:
        result=dict(pool.map(encode,bank("fit")))
    write(OUT/"encoded.json",result)
    print("gated examples",sum(r["n"] for r in result.values()),flush=True)


def train():
    torch.set_num_threads(2);torch.manual_seed(SEED+612)
    data={k:[] for k in FIELDS}
    for name,meta in read(OUT/"encoded.json").items():
        if not meta["n"]:continue
        file=OUT/"encoded"/(name+".npz");assert digest(file)==meta["sha256"]
        with np.load(file) as arrays:
            for k in data:data[k].append(arrays[k])
    d={k:np.concatenate(v) for k,v in data.items()}
    n=len(d["own"]);bad=~d["viable"][np.arange(n),d["own"]]&d["viable"].any(1)
    danger=(~d["viable"]&d["legal"]).any(1)
    groups=[np.flatnonzero(g) for g in (bad,(~bad)&danger&d["viable"].any(1),d["viable"].any(1),d["tier"]>=3)]
    assert all(map(len,groups))
    rng=np.random.default_rng(SEED+614)
    net=Correction(d["x"].shape[-1]);opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.008)
    losses=[];idx=torch.arange(128)
    for step in range(1000):
        ids=np.concatenate([rng.choice(g,n) for g,n in zip(groups,(24,40,40,24))])
        b={k:torch.from_numpy(a[ids]) for k,a in d.items()};legal=b["legal"];viable=b["viable"]
        z=net(b["x"].float(),legal,b["own"])
        viability=F.binary_cross_entropy_with_logits(z[:,:,0][legal],viable[legal].float())
        best=F.binary_cross_entropy_with_logits(z[:,:,1][legal],b["best"][legal].float())
        signs=z[:,:,2:].reshape(128,8,2,3)
        sensor=F.cross_entropy(signs[legal].reshape(-1,3),b["sign"][legal].reshape(-1))
        risk=z[:,:,0];alternative=(~viable)&legal
        # Жизнеспособные ходы должны стоять выше необратимых при общем наблюдении.
        pair=viable[:,:,None]&alternative[:,None,:]
        rank=F.softplus(1-risk[:,:,None]+risk[:,None,:])[pair].mean()
        loss=viability+.45*best+.8*sensor+.5*rank
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if step%100==99:losses.append(float(loss.detach()))
    assert digest(LATE)==LATE_SHA
    base=HERE/"weights/wave1.pt";pack=dict(format="cautious-prev-two-v1",dim=d["x"].shape[-1],
          correction=net.state_dict(),early=torch.load(base,weights_only=True,map_location="cpu"),
          late=torch.load(LATE,weights_only=True,map_location="cpu"),threshold=2.)
    path=OUT/"candidate.pt";torch.save(pack,path)
    write(OUT/"training.json",dict(weights=digest(path),wave1=digest(base),joint400=LATE_SHA,samples=n,
          pools=[len(g) for g in groups],steps=1000,loss=losses,encoded=digest(OUT/"encoded.json")))
    print("trained cautious",n,[len(g) for g in groups],flush=True)


if __name__=="__main__":
    if sys.argv[1]=="collect":collect()
    else:train()
