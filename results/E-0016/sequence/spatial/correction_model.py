"""Сопоставление прогнозов ранней развилки с ошибками собственного контроллера."""
from io import BytesIO
import numpy as np
import torch
from torch import nn
from maze.grid import cell,locations
from maze.interface import Move
from .prefix import Prefix
from .transfer_model import FullModel

MOVES=tuple(Move)
DIM=8+4+11+5


class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.move=nn.Sequential(nn.Linear(DIM,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.classifier=nn.Sequential(nn.Linear(64,40),nn.SiLU(),nn.Linear(40,3))

    def forward(self,x,legal):
        h=self.move(x);ctx=h.masked_fill(~legal[:,:,None],-1e5).amax(1,keepdim=True).expand_as(h)
        return self.classifier(torch.cat((h,ctx),-1))


class Evidence:
    def __init__(self,pack):
        self.net=FullModel(pack)
        self.bank=Prefix(pack["prefix"].numpy().view(np.uint32))

    def directions(self,v):
        old,ok=self.net.directions(v)
        x=np.zeros((8,DIM),np.float32)
        x[:,:8]=old.astype(np.float32)
        x[:,0]=np.clip((old[:,0]-old[ok,0].max())/6,-6,0)
        if not old[:,2:].any():return old[:,0],None,ok
        x[:,8:12]=self.bank.score(v)
        s=[v.fuel-v.fuel_distance,v.health-v.repair_distance]
        k=int(s[1]<s[0]);dx,dy=v.dx,v.dy
        measures=(v.fuel_distance,v.repair_distance)
        stock=(v.fuel,v.health)
        known=[]
        for symbol in ("F","R"):
            seen=0
            for i,j in locations(v.memory,symbol):
                seen=max(seen,sum(cell(v.memory,i+u,j+t)!="?" for u in (-1,0,1) for t in (-1,0,1)))
            known.append(seen)
        x[:,12:23]=[np.clip(s[k],-10,30)/10,np.clip(s[1-k],-10,50)/20,measures[k]/40,measures[1-k]/40,
                    stock[k]/100,stock[1-k]/100,known[k]/9,known[1-k]/9,ok.sum()/8,
                    max(abs(dx),abs(dy))/35,sum(line.count("?") for line in v.memory)/(len(v.memory)*len(v.memory[0]))]
        back=(-v.last_move.value[0],-v.last_move.value[1]) if v.last_move else None
        px,py=v.position
        for j,m in enumerate(MOVES):
            u,w=m.value
            fresh=sum(cell(v.memory,px+u+i,py+w+t)=="?" for i in (-1,0,1) for t in (-1,0,1))
            x[j,23:]=[m==v.last_move,(u,w)==back,fresh/6,abs(u*w),
                      max(abs(dx),abs(dy))-max(abs(dx-u),abs(dy-w))]
        x[~ok]=0
        return old[:,0],x,ok


class Correction:
    def __init__(self,pack):
        self.features=Evidence(pack)
        self.net=Calibrator().eval().requires_grad_(False)
        self.net.load_state_dict(pack["calibrator"])

    @torch.inference_mode()
    def directions(self,v):
        original,x,ok=self.features.directions(v)
        out=np.zeros((8,8),np.float64)
        out[:,0]=original
        if x is None:return out,ok
        q=self.net(torch.from_numpy(x[None]),torch.from_numpy(ok[None]))[0].sigmoid().numpy()
        out[:,1]=x[:,1]
        out[:,2:5]=q
        out[:,5]=x[:,8:10].mean(1)
        out[:,6]=x[:,5]
        out[:,7]=x[:,6]
        return out,ok


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=Correction(pack);w=pack["linear"].numpy()
    class Controller:
        def choose(self,v):
            z,ok=net.directions(v)
            return MOVES[int(np.where(ok,z@w,-1e12).argmax())]
    return Controller()
