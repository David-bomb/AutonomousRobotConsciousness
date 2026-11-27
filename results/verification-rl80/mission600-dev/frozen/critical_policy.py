"""Дополнительная обучаемая оценка жизнеспособности каждого физического хода."""
import torch
from torch import nn
from torch.nn import functional as F
from .policy import Policy


class CriticalPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.risk=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.zeros_(self.risk[-1].weight)
        nn.init.zeros_(self.risk[-1].bias)

    def heads(self,x,legal):
        sites=self.site(x[:,:,:110].reshape(*x.shape[:2],5,22))
        hidden=self.direction(torch.cat((sites[:,:,0],sites[:,:,1:3].amax(2),
                                          sites[:,:,3:5].amax(2),x[:,:,110:]),-1))
        context=hidden.masked_fill(~legal[:,:,None],-1e6).amax(1,keepdim=True).expand_as(hidden)
        both=torch.cat((hidden,context),-1)
        safety=self.risk(both).squeeze(-1)
        logits=self.score(both).squeeze(-1)+2*F.logsigmoid(safety)
        return logits.masked_fill(~legal,-1e6),safety

    def forward(self,x,legal):
        return self.heads(x,legal)[0]
