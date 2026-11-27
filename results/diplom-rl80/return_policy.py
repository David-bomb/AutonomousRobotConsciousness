"""One eight-action controller with frozen learned weights and the two previous real readings."""
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
prev=import_module("results.E-0016.prev-two.adapter")
base=import_module("results.E-0016.policy")
from maze.grid import destination

WEIGHTS=PROJECT/"results/E-0016/prev-two/weights/super1500.pt"
WEIGHTS_HASH="2669a75798b386c89da21c0bc16f558a6c57772b6ff736f453d14c060ec11e74"


def rank(column,legal):
    valid=(column<1.e7)&legal
    if not valid.any():return np.zeros(8)
    delta=column[valid]-column[valid].min()
    out=np.zeros(8)
    out[valid]=np.exp(-.85*delta)
    return out


class ReturnPolicy:
    def __init__(self,parameters=None):
        self.inner=prev.Controller(WEIGHTS,WEIGHTS_HASH).net
        self.lib=ctypes.CDLL(str(HERE/"late_frontiers.dll"))
        ci=ctypes.c_int
        self.lib.frontiers.argtypes=[ctypes.c_char_p,ci,ci,ci,np.ctypeslib.ndpointer(np.int32),ci,ci,
                                     np.ctypeslib.ndpointer(np.float64)]
        self.lib.frontiers.restype=None
        self.previous=(None,None)
        if parameters is not None:
            frozen=json.loads(Path(parameters).read_bytes())
            self.coefficients=np.asarray(frozen["coefficients"],np.float64)
            if len(self.coefficients)!=9 or frozen["base_sha256"]!=WEIGHTS_HASH:
                raise ValueError("Unexpected frozen parameters")

    def observed(self,v):
        legal=np.asarray([destination(v.window,(1,1),m) is not None for m in base.MOVES])
        features=np.zeros((8,9))
        if legal.sum()==1:
            return None,legal,features
        core,grad,checked=self.inner.compressed(v)
        assert np.array_equal(legal,checked)
        x,y=v.position
        h,w=len(v.memory),len(v.memory[0])
        sites,raw=base.geometry(v.memory,v.position,(x+v.dx,y+v.dy))
        ids=np.asarray(sites[:,1]*w+sites[:,0],dtype=np.int32)
        values=np.empty((8,9))
        self.lib.frontiers("".join(v.memory).encode("ascii"),w,h,y*w+x,ids,v.fuel,v.health,values)
        features[:,0]=rank(values[:,0],legal)
        if not features[:,0].any():
            safe=bool((values[:,1]<1.e7).any())
            if safe:
                for j in range(1,4):
                    features[:,j]=rank(values[:,j],legal)
            else:
                roads=raw[:,:15].reshape(8,5,3)[:,:,0]
                paired=roads[:,1:3].min()<250 and roads[:,3:].min()<250
                if paired and max(v.fuel,v.health)>45:
                    for j in range(4,9):
                        features[:,j]=rank(values[:,j],legal)
        return (core,grad), legal, features

    def physical_scores(self,v,observed,previous):
        data,legal,features=observed
        if data is None:return np.where(legal,0.,-1.e6),features
        core,grad=data
        past,possible=prev.memory_fields(v,previous,legal)
        with torch.inference_mode():
            q=self.inner(*prev.tensor_batch([(core,grad,past,possible,legal)]))[0].numpy()
        normalized=(q-q[legal].max())/max(float(q[legal].std()),.5)
        return normalized,features

    def choose(self,v):
        if v.last_move is None:self.previous=(None,None)
        scores,features=self.physical_scores(v,self.observed(v),self.previous)
        a=(scores+features@self.coefficients).argmax()
        self.previous=(v.fuel_distance,v.repair_distance)
        return base.MOVES[a]
