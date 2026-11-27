"""Learn action scores after a service-window handover, using only the robot's ordinary input."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from renewable_policy import targets
from service_window_policy import Mixture,coverage


class FinalLeg(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.residual=pack["residual"].numpy()
        self.lib=ctypes.CDLL(str(Path(__file__).with_name("proper_frontier.dll")))
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.lib.proper.argtypes=[ctypes.c_char_p,ci,ci,ci,ints,ci,ci,ints]
        self.lib.proper.restype=None

    def late_parts(self,v,prior):
        x,ok=self.b.directions(v)
        q=x@self.linear
        q=(q-q[ok].max())/max(.5,float(q[ok].std()))
        q[~ok]=-1.e6
        fields=np.zeros((8,9))
        if ok.sum()==1:return q,fields
        w=len(v.memory[0]);px,py=v.position
        ids=targets(v.memory,(px+v.dx,py+v.dy))
        mask=np.zeros(7,np.int32)
        self.lib.proper("".join(v.memory).encode("ascii"),w,len(v.memory),py*w+px,ids,v.fuel,v.health,mask)
        project=lambda m:np.array([(m>>i)&1 for i in range(8)],float)
        if mask[0]:fields[:,0]=project(mask[0]);return q,fields
        if mask[1]:
            fields[:,1]=project(mask[2])
            fields[:,2]=project(mask[1])
        else:
            fields[:,3]=project(mask[4])
            fields[:,4]=project(mask[6])
        if x[:,1].any() or x[:,2].any():return q,fields
        fields[:,5]=x[:,3]-x[:,4]
        fields[:,6]=x[:,3]+x[:,4]
        slack=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
        if slack<=12:
            early=self.early_scores(v,prior)
            early=(early-early[ok].max())/max(.5,float(early[ok].std()))
            fields[:,7]=np.clip(early,-4,0)*(slack<=3)
            fields[:,8]=np.clip(early,-4,0)*(3<slack<=12)
        return q,fields

    def choose(self,v,cache=None):
        if v.last_move is None:self.previous=(None,None)
        prior=(None,None) if self.ablate else self.previous
        key=v,prior
        if cache is None or key not in cache:
            if min(coverage(v.memory))==9:
                q,f=self.late_parts(v,prior)
                q=q+f@self.residual
            else:q=self.early_scores(v,prior)
            move=self.early.MOVES[int(q.argmax())]
            if cache is not None:cache[key]=move
        else:move=cache[key]
        self.previous=(v.fuel_distance,v.repair_distance)
        return move


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="full-window-final-leg-1":raise ValueError("Unexpected model format")
    torch.set_num_threads(1)
    return FinalLeg(pack,ablate)
