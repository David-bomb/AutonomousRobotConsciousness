"""A single frozen mixture. Routing depends only on the revealed 3x3 service windows."""
from functools import lru_cache
from importlib import import_module
from io import BytesIO

import numpy as np
import torch


@lru_cache(maxsize=512)
def coverage(memory):
    h,w=len(memory),len(memory[0])
    observed=[0,0]
    for y,row in enumerate(memory):
        for x,c in enumerate(row):
            if c not in "FR":continue
            n=sum(memory[j][i]!="?" for j in range(max(0,y-1),min(y+2,h))
                  for i in range(max(0,x-1),min(x+2,w)))
            s=int(c=="R");observed[s]=max(observed[s],n)
    return tuple(observed)


class Mixture:
    def __init__(self,pack,ablate=False):
        self.early=import_module("results.E-0016.prev-two.adapter")
        late=import_module("results.E-0016.sequence.belief.joint_model")
        self.a=self.early.PrevNet("plain").eval().requires_grad_(False)
        self.a.load_state_dict(pack["early"])
        self.b=late.Joint(pack["late"])
        self.linear=pack["late"]["linear"].numpy()
        self.cfg=pack["thresholds"].tolist()
        self.ablate=ablate;self.previous=(None,None)

    @torch.inference_mode()
    def early_scores(self,v,prior):
        pack=self.a.features(v,prior)
        return self.a(*self.early.tensor_batch([pack]))[0].numpy()

    def late_scores(self,v):
        x,legal=self.b.directions(v)
        return np.where(legal,x@self.linear,-1.e12)

    def route(self,v,cfg):
        seen=coverage(v.memory)
        least,greatest=sorted(seen)
        minimum,best,margin=cfg
        return (least>=minimum and greatest>=best and
                min(v.fuel-v.fuel_distance,v.health-v.repair_distance)>=margin)

    def choose(self,v,cache=None):
        if v.last_move is None:self.previous=(None,None)
        prior=(None,None) if self.ablate else self.previous
        key=(v,prior)
        if cache is None or key not in cache:
            q=self.late_scores(v) if self.route(v,self.cfg) else self.early_scores(v,prior)
            move=self.early.MOVES[int(q.argmax())]
            if cache is not None:cache[key]=move
        else:move=cache[key]
        self.previous=(v.fuel_distance,v.repair_distance)
        return move


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),map_location="cpu",weights_only=True)
    if pack["format"]!="service-window-mixture-1":raise ValueError("Unexpected model format")
    torch.set_num_threads(1)
    return Mixture(pack,ablate)
