"""Слой продолжения. Для выбора доступны только прежние входы prev-two."""
from io import BytesIO
import torch
from torch import nn
from .route_adapter import RouteNet
from .adapter import Controller as BasicController


class PatchNet(RouteNet):
    def __init__(self,kind):
        super().__init__(kind)
        self.patch_features=nn.Sequential(nn.Linear(20+196+2*4,80),nn.LayerNorm(80),nn.SiLU(),
                                          nn.Linear(80,48),nn.SiLU())
        self.patch_aux=nn.Linear(48,3)
        self.patch_score=nn.Sequential(nn.Linear(99,48),nn.SiLU(),nn.Linear(48,1))
        nn.init.zeros_(self.patch_score[-1].weight);nn.init.zeros_(self.patch_score[-1].bias)
        self.mix=nn.Parameter(torch.cat((self.mix.detach(),torch.ones(1))))

    def augmented(self,core,grad,past,possible,legal):
        base,signs,risk=super().parts(core,grad,past,possible,legal)
        s=past[:,:,:,[3,10,11,12]].flatten(2)
        edge=self.patch_features(torch.cat((base,core[:,:,1:197],s),-1))
        context=edge.masked_fill(~legal[:,:,None],-1e5).amax(1,keepdim=True).expand_as(edge)
        aux=self.patch_aux(edge)
        correction=self.patch_score(torch.cat((edge,context,aux.sigmoid()),-1))
        known=core[:,:,1:41].reshape(*core.shape[:2],5,8)
        proof=((known[:,:,:,6]>.5)&(known[:,:,:,2]>=0)&legal[:,:,None]).any(2).any(1,keepdim=True)
        safe=(core[:,:,194]>.5)&legal  # За границей обзора достижимы оба типа сервиса.
        room=(past[:,:,:,10].amin(2)>.22)&(past[:,:,:,12].amin(2)>.6)
        gate=proof|safe.any(1,keepdim=True)|room
        return torch.cat((base,correction*gate[:,:,None]),-1),signs,risk,aux

    def parts(self,*args):
        return self.augmented(*args)[:3]


class Controller(BasicController):
    def _load(self,data,ablate):
        state=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
        self.net=PatchNet("rich" if "base.base.wall_prior" in state else
                          "critical" if "base.base.risk.0.weight" in state else "plain")
        self.net.load_state_dict(state)
        self.net.requires_grad_(False).eval()
        self.ablate=ablate
        self.reset()


def from_bytes(data,ablate=False):
    obj=Controller.__new__(Controller)
    obj._load(data,ablate)
    return obj
