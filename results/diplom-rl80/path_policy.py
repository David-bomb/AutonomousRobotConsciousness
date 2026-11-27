"""Use train-fitted shortest-path transition probabilities inside the observed road graph."""
import ctypes
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from protected_policy import Protected
from renewable_policy import targets


class PathFields:
    def __init__(self,pack):
        self.table=pack["path_counts"].numpy()
        self.prior=pack["path_lengths"].numpy()
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32);floats=np.ctypeslib.ndpointer(np.float32)
        self.native=ctypes.CDLL(str(Path(__file__).with_name("learned_paths.dll"))).infer_paths
        self.native.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,floats,floats,np.ctypeslib.ndpointer(np.float64)]
        self.native.restype=None

    def get(self,v,past,k):
        w=len(v.memory[0]);x,y=v.position;p=y*w+x;previous=p
        if v.moved:
            dx,dy=v.last_move.value;previous-=dx+w*dy
        now=np.asarray([v.fuel_distance,v.repair_distance],np.int32)
        old=np.asarray([-1 if z is None else z for z in past],np.int32)
        out=np.zeros((8,6))
        self.native("".join(v.memory).encode(),w,len(v.memory),p,previous,targets(v.memory,(x+v.dx,y+v.dy)),
                    now,old,k,self.table,self.prior,out)
        if not np.isfinite(out).all():raise ValueError("Nonfinite route estimates")
        return out


class PathNet(Protected):
    def __init__(self,p,ablate=False):
        super().__init__(p,ablate)
        self.paths=PathFields(p)
        self.theta=p["paths"].numpy()

    def early_parts(self,v,past):
        q,k,slack,protected=self.foundation(v,past);x=np.zeros((8,12))
        if protected or not 0<=slack<=6:return q,x
        data=self.a.features(v,past);legal=data[-1];idx=1-k if v.health<v.fuel else k
        possible=legal&data[3][:,idx,0]
        if possible.sum()<2:return q,x
        parent=int(q.argmax())
        z=self.paths.get(v,past,k)
        x[:,:6]=z-z[parent,:]
        for j in range(6):
            best=int(np.where(possible,z[:,j],-1).argmax())
            if z[best,j]>.58 and z[best,j]-z[parent,j]>.32:
                x[best,j+6]=q[parent]-q[best]+1
        return q,x

    def early_scores(self,v,past):
        if not self.theta.any():return self.foundation(v,past)[0]
        q,x=self.early_parts(v,past)
        return q+x@self.theta


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="learned-paths-1":raise ValueError("Unknown policy format")
    return PathNet(p,ablate)
