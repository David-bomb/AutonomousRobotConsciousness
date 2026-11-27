from io import BytesIO
import numpy as np
import torch
from torch import nn
from ..belief.joint_model import MOVES
from .gate_model import GateNet
from .gates import describe
from .local_model import LocalNet,local_image
from .model import image
from .transfer_model import FullModel


class OutcomeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(28,36),nn.LayerNorm(36),nn.SiLU(),nn.Linear(36,24),nn.SiLU())
        self.score=nn.Sequential(nn.Linear(48,24),nn.SiLU(),nn.Linear(24,3))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self,x,legal):
        h=self.edge(x)
        c=h.masked_fill(~legal[...,None],-1e4).amax(1,keepdim=True).expand_as(h)
        return self.score(torch.cat((h,c),-1))


class Signals:
    def __init__(self,pack):
        self.gate=GateNet().eval().requires_grad_(False)
        self.gate.load_state_dict(pack["gate"])
        self.local=LocalNet().eval().requires_grad_(False)
        self.local.load_state_dict(pack["local"])

    @torch.inference_mode()
    def __call__(self,gates,patch,meta,stock,last,scores,legal):
        g,q=self.gate(gates.float())
        prob=q.sigmoid()
        slope,value,mask=self.local(patch,meta,stock,last)
        assert torch.equal(mask,legal)
        slip=stock.float()-meta[:,6:]
        u=slip.argmin(1)
        b=torch.arange(len(q))
        v=[v for v in prob.unbind(-1)]
        v.extend(slope.softmax(-1)[b,:,u].unbind(-1))
        v.extend(value.sigmoid().unbind(-1))
        p=g.sigmoid()*gates[...,0]
        v.extend((p.amax(2),p.sum(2).log1p(),gates[...,0].sum(2)/8))
        centered=scores-scores.masked_fill(~legal,-1e5).amax(1,keepdim=True)
        v.extend((centered.clamp(-20,0)/4,torch.exp(centered.clamp_max(0)),(centered==0).float()))
        dirs=torch.tensor([m.value for m in MOVES]).float()
        loc=torch.stack((meta[:,0]%40,meta[:,0]//40),1).float()
        goal=torch.stack((meta[:,5]%40,meta[:,5]//40),1).float()-loc
        a=torch.arange(8)[None]
        inv=torch.tensor([2,3,0,1,6,7,4,5,8])
        v.extend(((last[:,None]==a).float(),(inv[last.long()][:,None]==a).float(),(goal@dirs.T)/35))
        global_=(slip[b,u].clamp(-10,35)/15,slip[b,1-u].clamp(-10,50)/25,stock[b,u]/100,
                 stock[b,1-u]/100,meta[b,6+u]/40,meta[b,7-u]/40,legal.sum(1)/8,
                 goal.abs().amax(1)/35,
                 (stock>75).any(1).float(),gates[...,1].amax((1,2)),gates[...,0].sum((1,2))/36)
        v.extend(x[:,None].expand(-1,8) for x in global_)
        return torch.stack(v,-1).float()


class TerminalModel:
    def __init__(self,pack):
        self.base=FullModel(pack["foundation"])
        self.w=pack["foundation"]["linear"].numpy()
        self.sensors=Signals(pack)
        self.head=OutcomeHead().eval().requires_grad_(False)
        self.head.load_state_dict(pack["head"])

    @torch.inference_mode()
    def directions(self,view):
        prev,legal=self.base.directions(view)
        orig=prev@self.w
        cols=np.zeros((8,3),np.float64)
        cols[:,0]=orig-orig[legal].max()
        if legal.sum()<2:return cols,legal
        board,meta=image(view)
        stock=np.array([view.fuel,view.health],np.int16)
        gates,_,_=describe(board,meta,stock)
        patch,_=local_image(view)
        last=MOVES.index(view.last_move) if view.last_move else 8
        data=(gates,patch,meta,stock,np.asarray(last),orig.astype(np.float32),legal)
        inputs=[torch.from_numpy(i[None]) for i in data]
        feature=self.sensors(*inputs)
        q=self.head(feature,inputs[6])[0]
        cols[:,1]=q[:,0].clamp(-4,4).numpy()
        cols[:,2]=q[:,1].sigmoid().numpy()
        return cols,legal


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=TerminalModel(pack)
    weights=pack["linear"].numpy()

    class Controller:
        def choose(self,view):
            x,ok=net.directions(view)
            return MOVES[int(np.where(ok,x@weights,-1e12).argmax())]

    return Controller()
