"""Наблюдаемые выходы к неизученной области, совместная оценка сервиса и финиша."""
import ctypes as ct
from io import BytesIO
import numpy as np
import torch
from torch import nn
from maze.interface import Move
from ..policy import geometry
from ..rich_policy import estimated_memory
from .model import HERE,SequenceNet,encode


LIB=ct.CDLL(str(HERE/"alternatives.dll"))
LIB.alternatives.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.int32),
                         ct.c_int,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.float32)]
LIB.alternatives.restype=None


def exits(v,prior):
    px,py=v.position
    goal=(px+v.dx,py+v.dy)
    sites,_=geometry(v.memory,v.position,goal)
    estimated=estimated_memory(v.memory,goal,prior)
    f,h,df,dr=v.fuel,v.health,v.fuel_distance,v.repair_distance
    if h<f:
        sites=sites[[0,3,4,1,2]]
        f,h,df,dr=h,f,dr,df
    w=len(v.memory[0])
    ids=(sites[:,1]*w+sites[:,0]).astype(np.int32)
    features=np.zeros((8,5,22),np.float32)
    LIB.alternatives("".join(v.memory).encode(),"".join(estimated).encode(),w,len(v.memory),py*w+px,
                     ids,f,h,df,dr,features)
    return features


class ExitNet(SequenceNet):
    def __init__(self):
        super().__init__()
        self.exit_site=nn.Sequential(nn.Linear(22+49,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,40),nn.SiLU())
        self.exit_move=nn.Sequential(nn.Linear(40*3+10,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.exit_grad=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,6))
        self.exit_service=nn.Sequential(nn.Linear(196+64+12,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.exit_values=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,5))
        self.exit_gate=nn.Sequential(nn.Linear(64+12,32),nn.SiLU(),nn.Linear(32,5))
        self.exit_readout=nn.Linear(18,5,bias=False)
        self.exit_old=nn.Parameter(torch.tensor(.4))

    def full(self,x,other,mask):
        with torch.no_grad():
            old,old_grad=super().heads(x,mask)
        dims=(*x.shape[:2],5)
        sites=self.exit_site(torch.cat((other,x[:,:,:110].reshape(*dims,22),x[:,:,120:255].reshape(*dims,27)),-1))
        z=self.exit_move(torch.cat((sites[:,:,0],sites[:,:,1:3].amax(2),sites[:,:,3:].amax(2),x[:,:,110:120]),-1))
        grad=self.exit_grad(torch.cat((z,self.context(z,mask)),-1)).reshape(*x.shape[:2],2,3)
        state=self.exit_service(torch.cat((x[:,:,255:],z,grad.softmax(-1).flatten(2),old_grad.softmax(-1).flatten(2)),-1))
        context=self.context(state,mask)
        routes=x[:,:,255:295].reshape(*dims,8)
        route_length=routes[:,:,:,4].masked_fill(routes[:,:,:,6]<.5,3.).amin(2)
        front=x[:,:,415:427].reshape(*x.shape[:2],3,4)
        evidence=torch.cat((route_length[:,:,None],(route_length<2.4)[:,:,None],front.flatten(2),
                            grad.softmax(-1)[:,:,:,0],old_grad.softmax(-1)[:,:,:,0]),-1)
        low=evidence.masked_fill(~mask[:,:,None],4.).amin(1,keepdim=True)
        values=self.exit_values(torch.cat((state,context),-1))+self.exit_readout((evidence-low).clamp(-3,3))
        global_data=torch.cat((low[:,0,:2],low[:,0,2:14:4],low[:,0,3:14:4],
                               x[:,:,118:120].amax(1),x[:,:,[26,70]].masked_fill(~mask[:,:,None],4).amin(1)),1)
        gate=self.exit_gate(torch.cat((context[:,0],global_data),1))
        score=(values*gate.softmax(-1)[:,None,:]).sum(-1)+self.exit_old*old.masked_fill(~mask,0)
        return score.masked_fill(~mask,-1e6),grad,gate


def load(weights):
    state=torch.load(BytesIO(weights),weights_only=True,map_location="cpu")
    torch.set_num_threads(1)
    net=ExitNet().eval().requires_grad_(False)
    net.load_state_dict(state)
    prior=net.wall_prior.numpy().tobytes()
    class Controller:
        @torch.inference_mode()
        def choose(self,view):
            x,mask=encode(view,prior)
            extra=exits(view,prior)
            scores,_,_=net.full(torch.from_numpy(x[None]),torch.from_numpy(extra[None]),torch.from_numpy(mask[None]))
            return tuple(Move)[int(scores.argmax())]
    return Controller()
