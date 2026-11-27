"""Score the same observed fork in reflected and rotated coordinates."""
from dataclasses import replace
from functools import lru_cache
from io import BytesIO
import numpy as np
import torch
from maze.interface import Move
from service_window_policy import Mixture


def vector(x,y,t):
    if t&4:x,y=y,x
    return (-x if t&1 else x, -y if t&2 else y)


@lru_cache(maxsize=256)
def grid(rows,t):
    if t&4:rows=tuple("".join(col) for col in zip(*rows))
    if t&1:rows=tuple(row[::-1] for row in rows)
    if t&2:rows=rows[::-1]
    return rows


MOVES=tuple(Move)
PERM=np.asarray([[MOVES.index(Move(vector(*m.value,t))) for m in MOVES] for t in range(8)])


def transform(v,t):
    if not t:return v
    x,y=v.position;w,h=len(v.memory[0]),len(v.memory)
    if t&4:x,y,w,h=y,x,h,w
    if t&1:x=w-1-x
    if t&2:y=h-1-y
    dx,dy=vector(v.dx,v.dy,t)
    old=Move(vector(*v.last_move.value,t)) if v.last_move is not None else None
    return replace(v,position=(x,y),memory=grid(v.memory,t),window=grid(v.window,t),last_move=old,dx=dx,dy=dy)


class Symmetric(Mixture):
    @torch.inference_mode()
    def early_parts(self,v,old):
        original=self.a.features(v,old)
        legal=original[-1]
        q=self.a(*self.early.tensor_batch([original]))[0].numpy()
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        fields=np.zeros((8,7))
        ds=(v.fuel_distance,v.repair_distance);stock=(v.fuel,v.health)
        positive=[k for k in (0,1) if ds[k]>0]
        if legal.sum()<2 or not positive:return q,fields
        k=min(positive,key=lambda j:stock[j]-ds[j]);slack=stock[k]-ds[k]
        if slack>10:return q,fields
        priority=1-k if v.health<v.fuel else k
        possible=original[3][:,priority,0] & legal
        if possible.sum()<2:return q,fields
        batches=[original]+[self.a.features(transform(v,t),old) for t in range(1,8)]
        logits,sign,risk=self.a.heads(*self.early.tensor_batch(batches))
        aligned=np.take_along_axis(logits.numpy(),PERM,axis=1)
        assert all(np.array_equal(p[-1][PERM[t]],legal) for t,p in enumerate(batches))
        aligned=(aligned-aligned[:,legal].max(1)[:,None])/np.maximum(.5,aligned[:,legal].std(1))[:,None]
        sensor=np.take_along_axis(sign.softmax(-1)[:,:,priority,0].numpy(),PERM,axis=1)
        risk=np.take_along_axis(risk.sigmoid().numpy(),PERM,axis=1)
        base=aligned[:,legal].max(1,keepdims=True)
        votes=np.mean(np.isclose(aligned,base,atol=1.e-6),axis=0)
        fields[:,0]=aligned.mean(0)-q
        fields[:,1]=np.median(aligned,axis=0)-q
        fields[:,2]=votes
        fields[:,3]=sensor.mean(0)
        fields[:,4]=sensor.mean(0)*(slack<=2)
        fields[:,5]=risk.mean(0)
        fields[:,6]=possible*(slack<=1)
        fields[~legal]=0
        return q,fields

    def early_scores(self,v,old):
        q,x=self.early_parts(v,old)
        return q+x@self.residual


def load(blob,ablate=False):
    pack=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"]!="symmetry-window-1":raise ValueError("Unexpected checkpoint")
    net=Symmetric(pack,ablate)
    net.residual=pack["residual"].numpy()
    return net
