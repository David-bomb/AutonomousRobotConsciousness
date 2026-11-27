"""Чтение допустимых продолжений от двух реальных позиций при замороженной wave1."""
from io import BytesIO
from pathlib import Path
import ctypes as ct
import numpy as np
import torch
from . import cautious_policy as cp
from .early_policy import Controller as Pair

LIB=ct.CDLL(str(Path(__file__).with_name("path_counts.dll")))
LIB.path_counts.argtypes=[ct.c_char_p,*([ct.c_int]*4),cp.INTS,cp.INTS,cp.INTS,
                         np.ctypeslib.ndpointer(np.float64),np.ctypeslib.ndpointer(np.float64)]
LIB.path_counts.restype=None


def paths(v,previous,walls):
    x,y=v.position;w=len(v.memory[0]);old=y*w+x
    if v.moved:
        dx,dy=v.last_move.value;old-=dy*w+dx
    current=np.asarray((v.fuel_distance,v.repair_distance),np.int32)
    prior=np.asarray([-1 if p is None else p for p in previous],np.int32)
    out=np.zeros((8,2,2,6))
    LIB.path_counts("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,old,cp.sites(v.memory,(x+v.dx,y+v.dy)),
                    current,prior,np.asarray(walls,np.float64),out)
    return out


def design(v,previous,evidence,base,walls):
    x,gate=cp.design(v,previous,evidence,base)
    if not gate:return None,False
    raw=paths(v,previous,walls)
    k,_=cp.urgency(v);z=raw[:,[k,1-k]].copy()
    # Станционные адреса в типе симметричны, обобщение не зависит от порядка чтения сетки.
    counts=np.sort(z[:,:,:,:2],axis=-1);weights=np.sort(z[:,:,:,2:4],axis=-1)
    cols=[x]
    for a,scale in ((counts,12),(weights,12)):
        cols.append((a/scale).clip(0,4).reshape(8,-1))
        diff=a-a.max(0);cols.append((diff/5).clip(-4,0).reshape(8,-1))
        p=np.exp(diff);cols.append((p/np.maximum(p.sum(0),1e-10)).reshape(8,-1))
    cols.append(z[:,:,:,4:].sum(-1).reshape(8,-1)/2)
    return np.concatenate(cols,1).astype("f4"),True


def diagnostics(v,pair,walls,sign,k,legal):
    from collections import Counter
    urgent=list(pair);urgent[1-k]=None
    out={}
    for name,prev in (("both",pair),("urgent",urgent),("missing",(None,None))):
        raw=paths(v,prev,walls)[:,k]
        for mode,label in enumerate(("open","lattice")):
            x=raw[:,mode];available=(x[:,4:].sum(1)>0)&legal
            score=np.where(available,x[:,2:4].max(1),-1)
            out["paths_"+label+"_"+name]=Counter(states=1,candidates=int(available.sum()),
                retained=int((available&(sign==0)).sum()),true_down=int((legal&(sign==0)).sum()),
                top_down=int(available.any() and sign[score.argmax()]==0),no_candidates=int(not available.any()))
    return out


class Controller(cp.Controller):
    def __init__(self,pack,ablate):
        Pair.__init__(self,dict(format="prev-two-early-joint-v1",early=pack["early"],late=pack["late"]),ablate)
        self.ensemble=[]
        for state in pack["heads"]:
            net=cp.Correction(pack["dim"]);net.load_state_dict(state)
            self.ensemble.append(net.eval().requires_grad_(False))
        self.walls=np.asarray(pack["walls"],np.float64);self.threshold=float(pack["threshold"])

    @torch.inference_mode()
    def decision(self,v,previous):
        base=cp.base_parts(self.early,v,previous);q,legal,_,_=base;own=int(q.argmax())
        _,slack=cp.urgency(v)
        if legal.sum()<2 or slack<0 or slack>2:return own,own,0.
        x,gate=design(v,previous,cp.evidence(v,previous),base,self.walls)
        if not gate:return own,own,0.
        args=torch.from_numpy(x[None]),torch.from_numpy(legal[None]),torch.tensor([own])
        estimates=[]
        for net in self.ensemble:
            z=net(*args)[0];p=z[:,0].sigmoid()+.16*z[:,1].sigmoid()+.12*z[:,2:5].softmax(1)[:,0]
            estimates.append((p-p[own]).numpy())
        deltas=np.stack(estimates)
        advantage=.55*deltas.mean(0)+.45*deltas.min(0);advantage[~legal]=-5
        best=int(advantage.argmax());return own,best,float(advantage[best])


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="path-prev-two-v1":raise ValueError("Неизвестный формат")
    return Controller(pack,ablate)
