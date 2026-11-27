"""Дополнительное линейное чтение признаков дороги в той же сети prev-two."""
from io import BytesIO
import torch
from torch import nn
from .adapter import Controller as BasicController,PrevNet


class RouteNet(PrevNet):
    def __init__(self,kind):
        super().__init__(kind)
        self.register_buffer("route_parent_mix",self.mix.detach().clone())
        self.mix=nn.Parameter(torch.cat((self.route_parent_mix.clone(),torch.zeros(11))))

    def parts(self,core,grad,past,possible,legal):
        base,signs,risk=super().parts(core,grad,past,possible,legal)
        parent=base@self.route_parent_mix
        known=core[:,:,1:41].reshape(*core.shape[:2],5,8)
        feasible=(known[:,:,:,6]>.5)&(known[:,:,:,2]>=0)
        dist=known[:,:,:,4].masked_fill(~feasible,8).amin(2).masked_fill(~legal,8.)
        shortest=dist.amin(1,keepdim=True)
        confirmed=shortest<8
        front=core[:,:,161:173]*50
        valid=(core[:,:,185:197]>.5)&legal[:,:,None]

        def relative(index):
            d=front[:,:,index].masked_fill(~valid[:,:,index],300)
            best=d.amin(1,keepdim=True)
            return -(d-best).clamp(0,20),((best<300)&~confirmed),(d<=best+.001)

        safe2,has2,tie2=relative(9)
        safe1,has1,_=relative(5)
        any_front,has0,_=relative(1)
        goal_remaining,_,_=relative(10)
        nearest,_,_=relative(8)
        other=~has2&~confirmed
        room=past[:,:,:,12].amin(2)>.3
        fresh=core[:,:,235]  # Новое окно после первого хода в исходной геометрии.
        extra=torch.stack((parent*confirmed,parent*has2,
            (dist<=shortest+.00001)*confirmed,safe2*has2,tie2*has2,
            goal_remaining*has2,safe1*other*has1*room,any_front*other*~has1*has0*room,
            nearest*has2,fresh*has2,fresh*other*room),-1)
        return torch.cat((base,extra),-1),signs,risk


class Controller(BasicController):
    def _load(self,data,ablate):
        state=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
        self.net=RouteNet("rich" if "base.base.wall_prior" in state else
                          "critical" if "base.base.risk.0.weight" in state else "plain")
        self.net.load_state_dict(state)
        self.net.requires_grad_(False).eval()
        self.ablate=ablate
        self.reset()


def from_bytes(data,ablate=False):
    obj=Controller.__new__(Controller)
    obj._load(data,ablate)
    return obj
