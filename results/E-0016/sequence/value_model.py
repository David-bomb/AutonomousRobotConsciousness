"""Обученная на полных физических продолжениях оценка финиша по текущему наблюдению."""
from importlib import import_module
from io import BytesIO
import torch
from torch import nn
from .model import SequenceNet, encode


class ReturnNet(SequenceNet):
    def __init__(self):
        super().__init__()
        self.return_move=nn.Sequential(nn.Linear(451+7,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.return_score=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.return_score[-1].weight)
        nn.init.zeros_(self.return_score[-1].bias)
        self.register_buffer("return_strength",torch.tensor(1.))

    def values(self,x,legal):
        with torch.no_grad():
            base,grad=super().heads(x,legal)
        relative=(base-base.amax(1,keepdim=True)).clamp(-15,0)
        z=self.return_move(torch.cat((x,grad.softmax(-1).flatten(2),relative[:,:,None]/5),-1))
        val=self.return_score(torch.cat((z,self.context(z,legal)),-1)).squeeze(-1)
        return base,val

    def forward(self,x,legal):
        base,val=self.values(x,legal)
        return (base+self.return_strength*val).masked_fill(~legal,-1e6)


def load(weights):
    from maze.interface import Move
    state=torch.load(BytesIO(weights),weights_only=True,map_location="cpu")
    torch.set_num_threads(1)
    net=ReturnNet().eval().requires_grad_(False)
    net.load_state_dict(state)
    prior=net.wall_prior.numpy().tobytes()
    class Controller:
        @torch.inference_mode()
        def choose(self,view):
            x,ok=encode(view,prior)
            return tuple(Move)[int(net(torch.from_numpy(x[None]),torch.from_numpy(ok[None])).argmax())]
    return Controller()
