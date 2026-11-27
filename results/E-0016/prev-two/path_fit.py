"""Подгонка трёх независимых выходов по train-развилкам из десяти источников."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment
from . import early_data
from .adapter import MOVES
from .cautious_bank import OUT as FIRST,bank as first_bank
from .cautious_fit import FIELDS
from .cautious_policy import Correction,base_parts,evidence
from .common import HERE,SEED,digest,level,read,write
from .early_bank import LATE
from .path_bank import OUT,bank
from .path_policy import design

WALLS=None


def init():
    global WALLS
    early_data.init("wave1")
    WALLS=read(OUT/"registration.json")["wall_prior"]


def encode(job):
    row,folder=job;trip=read(folder/"prefix"/(row["name"]+".json"));env=Environment(level(row))
    events={r["step"]:r for r in trip["events"]};previous=(None,None);records=[]
    for step,c in enumerate(trip["actions"]):
        v=env.view
        if step in events:
            event=events[step];assert event["previous"]==list(previous)
            base=base_parts(early_data.NET,v,previous)
            x,gate=design(v,previous,evidence(v,previous),base,WALLS)
            if gate:
                k=event["urgent"];sign=np.asarray(event["signs"])[:,[k,1-k]]
                viable=np.array(event["viable"]);best=viable&(sign[:,0]==np.where(viable,sign[:,0],6).min())
                records.append((x.astype("f2"),base[1],event["own"],viable,sign,best,int(row["tier"][1]),
                                int(row["name"].split("-")[-3])))
        previous=v.fuel_distance,v.repair_distance;env.step(MOVES[int(c)])
    file=OUT/"encoded"/(row["name"]+".npz")
    if records:np.savez_compressed(file,**dict(zip(FIELDS,map(np.asarray,zip(*records)),strict=True)))
    return row["name"],dict(n=len(records),sha256=digest(file) if records else None)


def main():
    (OUT/"encoded").mkdir(exist_ok=True)
    jobs=[(r,FIRST) for r in first_bank("fit")]+[(r,OUT) for r in bank("fit")]
    with ProcessPoolExecutor(2,initializer=init) as pool:
        meta={}
        for name,item in pool.map(encode,jobs):
            meta[name]=item
            if len(meta)%250==0:print("geodesic features",len(meta),flush=True)
    write(OUT/"encoded.json",meta)
    d={k:[] for k in FIELDS}
    for n,record in meta.items():
        if record["n"]:
            with np.load(OUT/"encoded"/(n+".npz")) as a:
                for k in d:d[k].append(a[k])
    d={k:np.concatenate(v) for k,v in d.items()};N=len(d["own"]);index=np.arange(N)
    bad=~d["viable"][index,d["own"]]&d["viable"].any(1);danger=(~d["viable"]&d["legal"]).any(1)
    ids_seed=sorted(set(d["seed"]));heads=[];seeds=[]
    torch.set_num_threads(2)
    for i in range(3):
        seed=SEED+801+i;torch.manual_seed(seed);rng=np.random.default_rng(seed)
        # Каждый выход видит восемь или девять из десяти источников. Исключения фиксированы.
        keep=~np.isin(d["seed"],ids_seed[i::5])
        groups=[np.flatnonzero(keep&g) for g in (bad,~bad&danger&d["viable"].any(1),d["viable"].any(1),d["tier"]>=3)]
        net=Correction(d["x"].shape[-1]);opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.01)
        for step in range(500):
            ids=np.concatenate([rng.choice(g,n) for g,n in zip(groups,(24,40,40,24))])
            b={k:torch.from_numpy(v[ids]) for k,v in d.items()};legal=b["legal"];viable=b["viable"]
            z=net(b["x"].float(),legal,b["own"]);p=z[:,:,0]
            pair=viable[:,:,None]&((~viable)&legal)[:,None,:]
            loss=F.binary_cross_entropy_with_logits(p[legal],viable[legal].float())
            loss+=.5*F.softplus(1-p[:,:,None]+p[:,None,:])[pair].mean()
            loss+=.45*F.binary_cross_entropy_with_logits(z[:,:,1][legal],b["best"][legal].float())
            loss+=.9*F.cross_entropy(z[:,:,2:][legal].reshape(-1,3),b["sign"][legal].reshape(-1))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        heads.append(net.state_dict());seeds.append([int(s) for s in ids_seed if s not in ids_seed[i::5]])
    pack=dict(format="path-prev-two-v1",dim=d["x"].shape[-1],heads=heads,walls=read(OUT/"registration.json")["wall_prior"],
              early=torch.load(HERE/"weights/wave1.pt",map_location="cpu",weights_only=True),
              late=torch.load(LATE,map_location="cpu",weights_only=True),threshold=2.)
    file=OUT/"candidate.pt";torch.save(pack,file)
    write(OUT/"training.json",dict(sha256=digest(file),samples=N,harmful=int(bad.sum()),steps_per_head=500,
          sources_per_head=seeds,train_sha256=digest(OUT/"encoded.json")))
    print("geodesic trained",N,int(bad.sum()),flush=True)


if __name__=="__main__":main()
