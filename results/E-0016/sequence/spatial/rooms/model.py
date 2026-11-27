from io import BytesIO
import numpy as np
import torch
from torch import nn
from ..gate_model import GateNet
from ..gates import describe
from ..model import MOVES,image
from ..transfer_model import FullModel
from .cells import representation


class Family(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(32,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,4))

    def forward(self,x):return self.net(x)


class Action(nn.Module):
    def __init__(self):
        super().__init__()
        self.move=nn.Sequential(nn.Linear(66,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,48),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96,40),nn.SiLU(),nn.Linear(40,3))

    def forward(self,old,room,state,family,stock,meta,last,legal):
        a=torch.arange(8,device=old.device)[None]
        inv=torch.tensor([2,3,0,1,6,7,4,5,8],device=old.device)
        motion=torch.stack((last[:,None]==a,inv[last.long()][:,None]==a),-1).float()
        global_state=torch.cat((state,family,stock.float()/100,(stock-meta[:,6:]).float().clamp(-15,50)/25),1)
        x=torch.cat((old,room,room-old,motion,global_state[:,None].expand(-1,8,-1)),-1)
        h=self.move(x);ctx=h.masked_fill(~legal[:,:,None],-1e5).amax(1,keepdim=True).expand_as(h)
        return self.head(torch.cat((h,ctx),-1))


@torch.inference_mode()
def gate_channels(net,x):
    e,q=net(x.float())
    exists=x[...,0]>0
    p=e.sigmoid()*exists
    return torch.cat((q.sigmoid(),p.amax(2)[:,:,None],p.sum(2).log1p()[:,:,None],
         exists.float().sum(2)[:,:,None]/8,x[...,1].amax(2)[:,:,None],(exists*x[...,3]).amax(2)[:,:,None]),-1)


class RoomPolicy:
    def __init__(self,pack):
        self.base=FullModel(pack["foundation"])
        self.w=pack["foundation"]["linear"].numpy()
        self.gate=GateNet().eval().requires_grad_(False)
        self.family=Family().eval().requires_grad_(False)
        self.action=Action().eval().requires_grad_(False)
        for name in ("gate","family","action"):getattr(self,name).load_state_dict(pack[name])
        self.counts=pack["counts"].numpy()

    @torch.inference_mode()
    def directions(self,v):
        prior,ok=self.base.directions(v)
        z=np.zeros((8,10),np.float64);z[:,0]=prior@self.w
        if ok.sum()<2:return z,ok
        stock=np.array([v.fuel,v.health],np.int16)
        slack=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
        if slack>12:return z,ok
        board,m=image(v)
        hyp,state,added,blocked=representation(board,m,len(v.memory[0]),len(v.memory),self.counts)
        if hyp is board:return z,ok
        # Подтверждённое в действительном наблюдении завершение сохраняет свой выход.
        if not prior[:,2:].any():return z,ok
        old,_,_=describe(board,m,stock);after,_,_=describe(hyp,m,stock)
        raw=gate_channels(self.gate,torch.from_numpy(np.stack((old,after))))
        x,h=raw[:1],raw[1:]
        st=torch.from_numpy(state[None])
        fam=self.family(st).softmax(-1)
        args=(st,fam,torch.from_numpy(stock[None]),torch.from_numpy(m[None]),
              torch.tensor([MOVES.index(v.last_move) if v.last_move else 8]),torch.from_numpy(ok[None]))
        yes=self.action(x,h,*args)[0].sigmoid().numpy()
        no=self.action(x,x,*args)[0].sigmoid().numpy()
        p=float(fam[0,3]);weight=p*np.clip((13-slack)/6,0,1)
        z[:,1:4]=weight*(yes-no)
        z[:,4:6]=weight*(h[0,:,:2]-x[0,:,:2]).numpy()
        z[:,6]=weight*(h[0,:,6]-x[0,:,6]).numpy()
        if added.any():
            z[:,7:10]=weight*yes
        return z,ok


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=RoomPolicy(pack);w=pack["linear"].numpy()
    class Controller:
        def choose(self,v):
            z,ok=net.directions(v)
            return MOVES[int(np.where(ok,z@w,-1e12).argmax())]
    return Controller()
