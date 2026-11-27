"""Frozen eight-action scores from the two readings and exploration with renewable supplies."""
import ctypes
from functools import lru_cache
from importlib import import_module
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=128)
def targets(memory,goal):
    w=len(memory[0])
    sites=[goal[1]*w+goal[0]]
    for symbol in "FR":
        sites.extend(y*w+x for y,row in enumerate(memory) for x,ch in enumerate(row) if ch==symbol)
    if len(sites)!=5:
        raise ValueError("This controller expects the two fuel stations and two workshops of the stress task")
    return np.array(sites,dtype=np.int32)


def strength(cost):
    supported=cost<1.e7
    out=np.zeros(8)
    if supported.any():
        out[supported]=np.exp(-.9*(cost[supported]-cost[supported].min()))
    return out


class Controller:
    def __init__(self,core):
        self.api=import_module("results.E-0016.prev-two.adapter")
        self.net=self.api.from_bytes(core).net
        self.dll=ctypes.CDLL(str(HERE/"renewable_features.dll"))
        ci=ctypes.c_int
        self.dll.renewable.argtypes=[ctypes.c_char_p,ci,ci,ci,np.ctypeslib.ndpointer(np.int32),ci,ci,
                                     np.ctypeslib.ndpointer(np.float64)]
        self.dll.renewable.restype=None
        self.previous=(None,None)

    @torch.inference_mode()
    def scores(self,v,past):
        t=self.net.features(v,past)
        legal=t[-1]
        q=self.net(*self.api.tensor_batch([t]))[0].numpy()
        normalized=(q-q[legal].max())/max(.5,float(q[legal].std()))
        feats=np.zeros((8,7))
        if sum(legal)==1:return normalized,feats
        w=len(v.memory[0])
        x,y=v.position
        sites=targets(v.memory,(x+v.dx,y+v.dy))
        m=np.empty((8,7))
        self.dll.renewable("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,sites,v.fuel,v.health,m)
        if (m[:,0]<1.e7).any():
            feats[:,0]=strength(m[:,0])
        elif (m[:,1]<1.e7).any():
            for j in range(1,4):feats[:,j]=strength(m[:,j])
        elif max(v.fuel,v.health)>45 and (m[:,4]<1.e7).any():
            for j in range(4,7):feats[:,j]=strength(m[:,j])
        return normalized,feats

    def choose(self,v):
        if v.last_move is None:self.previous=(None,None)
        prior=(None,None) if self.ablate else self.previous
        q,feats=self.scores(v,prior)
        self.previous=(v.fuel_distance,v.repair_distance)
        return self.api.MOVES[int((q+feats@self.weights).argmax())]


def load(checkpoint,ablate=False):
    pack=torch.load(BytesIO(checkpoint),map_location="cpu",weights_only=True)
    if pack["format"]!="renewable-1":raise ValueError("Unknown format")
    core=BytesIO()
    torch.save(pack["core"],core)
    result=Controller(core.getvalue())
    result.weights=pack["weights"].numpy()
    result.ablate=ablate
    return result
