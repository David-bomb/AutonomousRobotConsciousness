"""Compare sampled geometry features with the frozen recovery800 control on fresh train cases."""
from collections import Counter
import ctypes
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.environment import Environment, Status
from maze.grid import destination

data=import_module("results.E-0016.data")
base=import_module("results.E-0016.policy")
teacher=import_module("results.E-0016.teacher")


def fit_morphology(rows):
    params=[]
    for family in ("corridors","branching","loops","rooms"):
        extra=remaining=plazas=total=0
        for r in rows:
            if r["family"]!=family:
                continue
            a=np.asarray([list(row) for row in r["grid"]])[1:-1,1:-1]!="#"
            nodes=a[::2,::2].size
            connectors=[a[::2,1::2],a[1::2,::2]]
            extra+=sum(s.sum() for s in connectors)-(nodes-1)
            remaining+=sum(s.size for s in connectors)-(nodes-1)
            plaza=a[1::2,1::2]
            plazas+=plaza.sum()
            total+=plaza.size
        params.extend([max(0.,extra/remaining),plazas/total])
    return np.asarray(params,np.float64)


class GeometryBelief:
    def __init__(self, params, particles):
        self.params=params
        self.particles=particles
        self.lib=ctypes.CDLL(str(HERE/"belief_worlds.dll"))
        ci=ctypes.c_int
        self.lib.worlds.argtypes=[ctypes.c_char_p,ci,ci,ci,np.ctypeslib.ndpointer(np.int32),ci,ci,ci,ci,
                                  np.ctypeslib.ndpointer(np.float64),ci,np.ctypeslib.ndpointer(np.float64)]
        self.lib.worlds.restype=None

    def features(self,v):
        x,y=v.position
        memory=v.memory
        h,w=len(memory),len(memory[0])
        sites,_=base.geometry(memory,v.position,(x+v.dx,y+v.dy))
        ids=np.asarray(sites[:,0]+w*sites[:,1],np.int32)
        out=np.zeros((8,6),np.float64)
        self.lib.worlds("".join(memory).encode("ascii"),w,h,y*w+x,ids,v.fuel_distance,v.repair_distance,
                        v.fuel,v.health,self.params,self.particles,out)
        return out


def main():
    torch.set_num_threads(1)
    exp=PROJECT/"results/E-0016"
    params=fit_morphology(data.rows(exp/"data/train.jsonl")[:600])
    estimator=GeometryBelief(params,96)
    model=base.Policy().eval()
    weights=exp/"runs/recovery800/rl.pt"
    model.load_state_dict(torch.load(weights,weights_only=True,map_location="cpu"))
    observations=[]
    for item in data.rows(exp/"sequence/train.jsonl")[1000:1100]:
        env=Environment(data.level(item))
        oracle=teacher.Teacher(env.level)
        seen=set()
        while env.status==Status.RUNNING and env.view not in seen:
            v=env.view
            seen.add(v)
            features,legal=base.features(v)
            with torch.inference_mode():
                logits=model(torch.from_numpy(features[None]),torch.from_numpy(legal[None]))[0].numpy()
            pilot=int(logits.argmax())
            which=min(range(2),key=lambda i:(v.fuel-v.fuel_distance,v.health-v.repair_distance)[i])
            slack=(v.fuel-v.fuel_distance,v.health-v.repair_distance)[which]
            distance=(v.fuel_distance,v.repair_distance)[which]
            start=env.steps==0
            tight=legal.sum()>1 and slack<=2 and distance>0
            if start or tight:
                stats=estimator.features(v)
                options=np.flatnonzero(legal).tolist()
                descent=max(options,key=lambda i:(stats[i,which],logits[i]))
                plan=max(options,key=lambda i:(stats[i,4],stats[i,which],logits[i]))
                closer=[]
                for i in options:
                    q=destination(v.memory,v.position,base.MOVES[i])
                    if min(d[q] for d in oracle.dist[2*which:2*which+2])<distance:closer.append(i)
                _,viable=oracle.labels(env)
                observations.append(dict(map=item["name"],step=env.steps,start=start,tight=bool(tight),
                                         pilot=pilot,descent=descent,plan=plan,closer=closer,viable=np.flatnonzero(viable).tolist()))
            if tight:break
            env.step(base.MOVES[pilot])
    groups={}
    for group in ("start","tight"):
        subset=[r for r in observations if r[group]]
        groups[group]=dict(n=len(subset))
        for outcome in ("closer","viable"):
            groups[group][outcome]={choice:sum(r[choice] in r[outcome] for r in subset) for choice in ("pilot","descent","plan")}
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (weights,HERE/"belief_worlds.cpp",HERE/"belief_worlds.dll")}
    payload=dict(groups=groups,fit="first 600 E-0016 train; four aggregate morphology priors",
                 test="E-0016 sequence/train[1000:1100]",particles=96,morphology=params.tolist(),hashes=hashes,records=observations)
    out=HERE/"belief-probe.json"
    encoded=(json.dumps(payload,ensure_ascii=False,indent=2)+"\n").encode()
    if out.exists() and out.read_bytes()!=encoded:raise ValueError("Previous measurement differs")
    out.write_bytes(encoded)
    print(json.dumps(groups))


if __name__=="__main__":
    main()
