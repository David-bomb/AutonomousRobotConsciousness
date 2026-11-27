"""Value an exploration move by where roads from remote targets first enter known terrain."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from departure_policy import Departure
from renewable_policy import targets


class BoundaryNet(Departure):
    def __init__(self,p,ablate=False):
        super().__init__(p,ablate)
        self.walk=ctypes.CDLL(str(Path(__file__).with_name("boundary_walk.dll"))).boundary_values
        ci=ctypes.c_int
        self.walk.argtypes=[ctypes.c_char_p,ci,ci,ci,np.ctypeslib.ndpointer(np.int32),ci,ci,
                            np.ctypeslib.ndpointer(np.float64)]
        self.walk.restype=None
        self.weights=p["boundary"].numpy()

    def late_parts(self,v,past):
        action,legal=self.b.directions(v)
        q=action@self.linear
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        q[~legal]=-1.e6;fields=np.zeros((8,12))
        if legal.sum()<2:return q,fields
        x,y=v.position;w,h=len(v.memory[0]),len(v.memory);pos=y*w+x
        ids=targets(v.memory,(x+v.dx,y+v.dy))
        m="".join(v.memory).encode();masks=np.empty(7,np.int32)
        self.known(m,w,h,pos,ids,v.fuel,v.health,masks)
        if masks[0] or masks[1] or not masks[4]:return q,fields
        self.walk(m,w,h,pos,ids,v.fuel,v.health,fields)
        assert np.isfinite(fields).all() and fields.min()>=0 and fields.max()<=1.000001
        return q,fields


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="boundary-walk-1":raise ValueError("Unknown policy format")
    return BoundaryNet(p,ablate)
