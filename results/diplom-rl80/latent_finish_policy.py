"""A terminal-trained correction for exploration beyond known safe service routes."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from renewable_policy import targets
from service_window_policy import Mixture


class LatentFinish(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        root=Path(__file__).resolve().parent
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.proper=ctypes.CDLL(str(root/"proper_frontier.dll"))
        self.proper.proper.argtypes=[ctypes.c_char_p,ci,ci,ci,ints,ci,ci,ints]
        self.proper.proper.restype=None
        self.worlds=ctypes.CDLL(str(root/"latent_finish.dll"))
        self.worlds.latent_finish.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,ci,ci,
                                            np.ctypeslib.ndpointer(np.float64)]
        self.worlds.latent_finish.restype=None
        self.coeff=pack["coeff"].numpy();self.samples=int(pack["samples"])

    def late_parts(self,v,past):
        scores,legal=self.b.directions(v);q=scores@self.linear
        q=(q-q[legal].max())/max(.5,float(q[legal].std()));q[~legal]=-1.e6
        out=np.zeros((8,7))
        if legal.sum()<2:return q,out
        w=len(v.memory[0]);x,y=v.position;p=y*w+x;h=len(v.memory)
        ids=targets(v.memory,(x+v.dx,y+v.dy));mem="".join(v.memory).encode("ascii")
        masks=np.zeros(7,np.int32)
        self.proper.proper(mem,w,h,p,ids,v.fuel,v.health,masks)
        def project(mask):return [(mask>>a)&1 for a in range(8)]
        if masks[0]:out[:,0]=project(masks[0]);return q,out
        if masks[1]:return q,out
        oldp=p
        if v.moved:
            dx,dy=v.last_move.value;oldp-=dx+dy*w
        old=np.array([-1 if s is None else s for s in past],np.int32)
        ds=np.array([v.fuel_distance,v.repair_distance],np.int32)
        fields=np.empty((8,5))
        self.worlds.latent_finish(mem,w,h,p,oldp,ids,ds,old,v.fuel,v.health,self.samples,fields)
        likelihood=fields[:,0].max()
        if likelihood>0:out[:,1:6]=fields/max(.10,float(likelihood))
        out[:,6]=project(masks[4])
        return q,out

    def choose(self,v,cache=None):
        if v.last_move is None:self.previous=(None,None)
        past=(None,None) if self.ablate else self.previous
        key=v,past
        if cache is None or key not in cache:
            if self.route(v,self.cfg):
                q,x=self.late_parts(v,past);q=q+x@self.coeff
            else:q=self.early_scores(v,past)
            action=self.early.MOVES[int(q.argmax())]
            if cache is not None:cache[key]=action
        else:action=cache[key]
        self.previous=(v.fuel_distance,v.repair_distance)
        return action


def load(blob,ablate=False):
    pack=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"]!="latent-finish-1":raise ValueError("Unknown model format")
    return LatentFinish(pack,ablate)
