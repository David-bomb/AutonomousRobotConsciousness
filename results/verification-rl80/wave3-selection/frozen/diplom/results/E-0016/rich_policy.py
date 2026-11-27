"""Обученный на train прогноз стен, используется как дополнительное чтение памяти."""
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .critical_policy import CriticalPolicy
from .policy import features, geometry, XY


@lru_cache(maxsize=12)
def contexts(h,w):
    y,x=np.indices((h,w))
    return (x%2 + 2*(y%2) + 4*((x==0)|(y==0)|(x==w-1)|(y==h-1))).astype(np.int32)


def context_codes(field):
    h,w=field.shape
    cells=np.where(field==ord("?"),0,np.where(field==ord("#"),1,2))
    padded=np.pad(cells,1,constant_values=1)
    return contexts(h,w)*81 + padded[:-2,1:-1]+3*padded[2:,1:-1]+9*padded[1:-1,:-2]+27*padded[1:-1,2:]


@lru_cache(maxsize=4096)
def estimated_memory(memory,goal,prior_bytes):
    board=np.asarray([list(r.encode("ascii")) for r in memory],np.uint8)
    prior=np.frombuffer(prior_bytes,np.float32)
    predicted=board.copy()
    hidden=board==ord("?")
    predicted[hidden & (prior[context_codes(board)]<.14)]=ord("#")
    if hidden[goal[1],goal[0]]:
        predicted[goal[1],goal[0]]=ord("?")
    return tuple(r.tobytes().decode("ascii") for r in predicted)


def inputs(view,prior_bytes):
    x,legal=features(view)
    pos=view.position
    goal=pos[0]+view.dx,pos[1]+view.dy
    predicted=estimated_memory(view.memory,goal,prior_bytes)
    sites,raw=geometry(predicted,pos,goal)
    _,observed=geometry(view.memory,pos,goal)
    original=observed[:,:15].reshape(8,5,3)
    paths=raw[:,:15].reshape(8,5,3)
    sensor=np.array([0,view.fuel_distance,view.fuel_distance,view.repair_distance,view.repair_distance],np.float32)
    stock=np.array([min(view.health,view.fuel),view.fuel,view.fuel,view.health,view.health],np.float32)
    rel=sites-np.asarray(pos)
    manhattan=np.abs(rel).sum(1)
    columns=[]
    def field(a):
        columns.append(np.broadcast_to(a,(8,5)))
    field(manhattan/60)
    field(manhattan-np.abs(rel[None,:,:]-XY[:,None,:]).sum(2))
    field(np.abs(sites-sites[0]).sum(1)/60)
    for d in (original[:,:,1],paths[:,:,1],paths[:,:,2]):
        field(np.minimum(d,120)/60)
        field(np.minimum(d-d.min(0),20)/5)
        field(np.clip(d-sensor,-24,24)/10)
        field(np.clip(d-stock,-24,24)/10)
        field(d<=sensor)
        plausible=(d<=sensor).sum(0)
        field(plausible/8)
        field((d<=sensor)&(plausible==1))
    field(original[:,:,0]==sensor)
    field(np.minimum(original[:,:,0],100)/50)
    field(np.clip(paths[:,:,1]-original[:,:,1],0,30)/10)
    active=np.stack(columns,-1).astype(np.float32)
    if view.health<view.fuel:
        active=active[:,[0,3,4,1,2]]
    out=np.concatenate((x,active.reshape(8,-1)),axis=1)
    out[~legal]=0
    return out,legal


class RichPolicy(CriticalPolicy):
    def __init__(self):
        super().__init__()
        self.register_buffer("wall_prior",torch.zeros(648))
        self.ext_site=nn.Sequential(nn.Linear(27+22,48),nn.ReLU(),nn.Linear(48,32),nn.ReLU())
        self.ext_direction=nn.Sequential(nn.Linear(32*3+10,64),nn.ReLU(),nn.Linear(64,48),nn.ReLU())
        self.ext_heads=nn.Sequential(nn.Linear(96,48),nn.ReLU(),nn.Linear(48,2))
        nn.init.zeros_(self.ext_heads[-1].weight)
        nn.init.zeros_(self.ext_heads[-1].bias)

    def heads(self,x,legal):
        old,safety=super().heads(x[:,:,:120],legal)
        fields=torch.cat((x[:,:,:110].reshape(*x.shape[:2],5,22),x[:,:,120:].reshape(*x.shape[:2],5,27)),-1)
        sites=self.ext_site(fields)
        hidden=self.ext_direction(torch.cat((sites[:,:,0],sites[:,:,1:3].amax(2),sites[:,:,3:5].amax(2),x[:,:,110:120]),-1))
        ctx=hidden.masked_fill(~legal[:,:,None],-1e6).amax(1,keepdim=True).expand_as(hidden)
        plus=self.ext_heads(torch.cat((hidden,ctx),-1))
        safe=safety+plus[:,:,1]
        return (old+plus[:,:,0]+2*(F.logsigmoid(safe)-F.logsigmoid(safety))).masked_fill(~legal,-1e6),safe
