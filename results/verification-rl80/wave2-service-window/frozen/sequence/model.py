"""Признаки миссии и прогноз показаний. Единственный вход решения — Observation."""
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..policy import Policy,geometry
from ..critical_policy import CriticalPolicy
from ..rich_policy import RichPolicy
from ..rich_policy import inputs, estimated_memory

HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/"sequence.dll"))
LIB.sequences.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.int32),
                        ct.c_int,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.float32)]
LIB.sequences.restype=None


def encode(v,prior):
    base,legal=inputs(v,prior)
    x,y=v.position
    goal=x+v.dx,y+v.dy
    estimates=estimated_memory(v.memory,goal,prior)
    sites,_=geometry(v.memory,v.position,goal)
    f,h,df,dr=v.fuel,v.health,v.fuel_distance,v.repair_distance
    if h<f:
        f,h,df,dr=h,f,dr,df
        sites=sites[[0,3,4,1,2]]
    w=len(v.memory[0])
    ids=(sites[:,1]*w+sites[:,0]).astype(np.int32)
    raw=np.zeros((8,92),np.float32)
    LIB.sequences("".join(v.memory).encode("ascii"),"".join(estimates).encode("ascii"),w,len(v.memory),y*w+x,ids,
                  f,h,df,dr,raw)
    paths=raw[:,:80].reshape(8,20,4)
    lengths,margins,feasible,count=paths.transpose(2,0,1)
    def relative(length):
        return np.clip(length-length[legal].min(0),0,20)/5
    legs=np.stack((np.minimum(lengths,180)/75,relative(lengths),np.clip(margins,-30,30)/15,
                   np.clip(margins[legal].max(0)-margins,0,20)/5,
                   np.minimum(feasible,180)/75,relative(feasible),(count>0),count/4),-1)
    frontier=raw[:,80:]
    info=np.concatenate((base,legs.reshape(8,-1),np.minimum(frontier,150)/50,
                         relative(frontier),frontier<300),axis=1).astype(np.float32)
    info[~legal]=0
    return info,legal  # 255 + 20*8 + 36 = 451.


class SequenceNet(nn.Module):
    def __init__(self,base_kind="plain"):
        super().__init__()
        self.register_buffer("wall_prior",torch.zeros(648))
        self.base=(RichPolicy() if base_kind=="rich" else CriticalPolicy() if base_kind=="critical" else Policy())
        self.sensor_site=nn.Sequential(nn.Linear(49,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.sensor_move=nn.Sequential(nn.Linear(32*3+10,80),nn.SiLU(),nn.Linear(80,64),nn.SiLU())
        self.sensor_out=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,6))
        self.mission=nn.Sequential(nn.Linear(196+64+6+1,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.action=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.action[-1].weight)
        nn.init.zeros_(self.action[-1].bias)

    @staticmethod
    def context(hidden,mask):
        return hidden.masked_fill(~mask[:,:,None],-1e5).amax(1,keepdim=True).expand_as(hidden)

    def sensor(self,x,mask):
        sites=torch.cat((x[:,:,:110].reshape(*x.shape[:2],5,22),x[:,:,120:255].reshape(*x.shape[:2],5,27)),-1)
        sites=self.sensor_site(sites)
        move=self.sensor_move(torch.cat((sites[:,:,0],sites[:,:,1:3].amax(2),sites[:,:,3:].amax(2),x[:,:,110:120]),-1))
        grad=self.sensor_out(torch.cat((move,self.context(move,mask)),-1)).reshape(*x.shape[:2],2,3)
        return move,grad

    def heads(self,x,mask):
        move,grad=self.sensor(x,mask)
        base=self.base(x[:,:,:255 if isinstance(self.base,RichPolicy) else 120],mask).masked_fill(~mask,0)
        per_action=self.mission(torch.cat((x[:,:,255:],move,grad.softmax(-1).flatten(2),base[:,:,None]/8),-1))
        residual=self.action(torch.cat((per_action,self.context(per_action,mask)),-1)).squeeze(-1)
        return (base+residual).masked_fill(~mask,-1e6),grad

    def forward(self,x,mask):
        return self.heads(x,mask)[0]
