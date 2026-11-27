"""Rank early alternatives using sampled terrain with exact joint sensor and mission constraints."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from origin_trace import origins
from protected_policy import Protected
from renewable_policy import targets


def args(v,past):
    w=len(v.memory[0]);x,y=v.position;pos=y*w+x;prior=pos
    if v.moved:
        dx,dy=v.last_move.value;prior-=dx+w*dy
    starts=np.asarray(origins(v),dtype=np.int32).reshape(-1,3)
    current=np.array([v.fuel_distance,v.repair_distance],np.int32)
    old=np.array([-1 if z is None else z for z in past],np.int32)
    return ("".join(v.memory).encode("ascii"),w,len(v.memory),pos,prior,targets(v.memory,(x+v.dx,y+v.dy)),
            current,old,v.fuel,v.health,starts,len(starts),int(v.last_move is None))


class Exact:
    def __init__(self,cap,stop):
        self.cap=cap;self.stop=stop;ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.dll=ctypes.CDLL(str(Path(__file__).with_name("exact_mission.dll")))
        self.dll.exact_votes.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,ci,ints,ci,ci,ci,ci,
                                       np.ctypeslib.ndpointer(np.float64),ints]
        self.dll.exact_votes.restype=None

    def get(self,v,past):
        out=np.empty((6,8,3));counts=np.empty(6,dtype=np.int32)
        self.dll.exact_votes(*args(v,past),self.cap,self.stop,out,counts)
        if not np.isfinite(out).all():raise ValueError("Nonfinite conditional estimates")
        return out,counts


class ExactNet(Protected):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.estimate=Exact(int(pack["cap"]),int(pack["stop"]))
        self.exact=pack["exact"].numpy()

    def early_parts(self,v,past):
        q,k,slack,protected=self.foundation(v,past);fields=np.zeros((8,12))
        if protected or not 0<=slack<=4:return q,fields
        p=self.a.features(v,past);idx=1-k if v.health<v.fuel else k
        ok=p[-1]&p[3][:,idx,0]
        if ok.sum()<2:return q,fields
        estimates,count=self.estimate.get(v,past)
        parent=int(q.argmax())
        # Back off from a sparse exact match by its number of independent proposals.
        exact_gauges=count[3]/(count[3]+14.);exact_mission=count[5]/(count[5]+9.)
        valid_witness=count[4]/(count[4]+20.)
        soft=estimates[1,:,k]
        conditional=(1-exact_gauges)*soft+exact_gauges*estimates[4,:,k]
        joint=(1-exact_mission)*conditional+exact_mission*estimates[5,:,k]
        mission=(1-valid_witness)*soft+valid_witness*estimates[2,:,k]
        w,h=len(v.memory[0])//2,len(v.memory)//2
        hard=max(w,h)>=15 and min(w,h)>=13
        for j,(belief,threshold,active) in enumerate(((conditional,.24,True),(conditional,.46,True),
            (joint,.24,True),(joint,.46,True),(joint,.66,True),(mission,.42,True),(joint,.30,hard),
            (joint,.40,slack<=1),(conditional,.35,hard),(soft,.40,True))):
            if not active:continue
            best=int(np.where(ok,belief,-1).argmax())
            if belief[best]-belief[parent]>threshold and belief[best]>.58:fields[best,j]=q[parent]-q[best]+1.
        fields[:,10]=joint-joint[parent];fields[:,11]=conditional-conditional[parent]
        return q,fields

    def early_scores(self,v,past):
        if not self.exact.any():return self.foundation(v,past)[0]
        q,x=self.early_parts(v,past)
        return q+x@self.exact


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="exact-mission-1":raise ValueError("Unknown format")
    return ExactNet(p,ablate)
