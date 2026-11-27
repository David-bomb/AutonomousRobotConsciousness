"""Conditional planning with held-out terrain samples and several observation horizons."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from protected_policy import Protected
from origin_trace import origins
from service_window_policy import coverage
from renewable_policy import targets


class HorizonFields:
    def __init__(self,samples):
        self.samples=samples;integer=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.native=ctypes.CDLL(str(Path(__file__).with_name("observation_tree.dll"))).observable_tree
        self.native.argtypes=[ctypes.c_char_p,integer,integer,integer,integer,ints,ints,ints,integer,integer,
                              ints,integer,integer,integer,np.ctypeslib.ndpointer(np.float64)]
        self.native.restype=None

    def estimate(self,v,past):
        w=len(v.memory[0]);x,y=v.position;p=y*w+x;oldp=p
        if v.moved:
            dx,dy=v.last_move.value;oldp-=dx+w*dy
        ids=targets(v.memory,(x+v.dx,y+v.dy))
        options=origins(v) if max(coverage(v.memory))<9 else ()
        starts=np.asarray(options,np.int32).reshape(-1,3)
        current=np.asarray([v.fuel_distance,v.repair_distance],np.int32)
        old=np.asarray([-1 if z is None else z for z in past],np.int32)
        out=np.zeros((8,6))
        self.native("".join(v.memory).encode(),w,len(v.memory),p,oldp,ids,current,old,v.fuel,v.health,starts,
                    len(starts),int(v.last_move is None),self.samples,out)
        assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1.600001
        return out


class Horizon(Protected):
    def __init__(self,p,ablate=False):
        super().__init__(p,ablate);self.predict=HorizonFields(int(p["horizon_samples"]))
        self.coeff=p["horizon"].numpy()

    def early_parts(self,v,past):
        q,k,slack,protect=self.foundation(v,past);x=np.zeros((8,12))
        if protect or not 0<=slack<=6:return q,x
        inputs=self.a.features(v,past);legal=inputs[-1];kind=1-k if v.health<v.fuel else k
        if (legal&inputs[3][:,kind,0]).sum()<2:return q,x
        parent=int(q.argmax());z=self.predict.estimate(v,past)
        if not z.any():return q,x
        x[:,:6]=z-z[parent,:]
        for j in (0,2,4):
            a=int(np.where(legal,z[:,j],-2).argmax())
            if z[a,j]-z[parent,j]>.12:x[a,6+j//2]=q[parent]-q[a]+1
            if z[a,j]-z[parent,j]>.20 and slack<=2:x[a,9+j//2]=q[parent]-q[a]+1
        return q,x

    def early_scores(self,v,past):
        if not self.coeff.any():return self.foundation(v,past)[0]
        q,x=self.early_parts(v,past);return q+x@self.coeff


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="cross-fit-horizon-1":raise ValueError("Unknown policy type")
    return Horizon(p,ablate)
