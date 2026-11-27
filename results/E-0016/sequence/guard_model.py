"""Обучаемое сопоставление направлений с порядком заездов и прогнозом остатка."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import SequenceNet


class GuardNet(SequenceNet):
    def __init__(self,kind="plain"):
        super().__init__(kind)
        self.route_embed=nn.Sequential(nn.Linear(32,24),nn.SiLU(),nn.Linear(24,16),nn.SiLU())
        self.patch_move=nn.Sequential(nn.Linear(5*16+36+6+2+2+1,96),nn.LayerNorm(96),nn.SiLU(),
                                      nn.Linear(96,64),nn.SiLU())
        self.phase_next=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))
        self.patch_scores=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
        self.risk_head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.patch_scores[-1].weight)
        nn.init.zeros_(self.patch_scores[-1].bias)
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.constant_(self.risk_head[-1].bias,2)

    def parts(self,x,mask):
        old,grad=super().heads(x,mask)
        obs=x[:,:,:110].reshape(*x.shape[:2],5,22)
        slack=obs[:,:,[1,3],4]*20
        probability=grad.softmax(-1)
        delta=torch.arange(-1,2,dtype=x.dtype,device=x.device)
        reserve=slack[:,:,:,None]-1-delta
        nonnegative=(probability*(reserve>=-.05)).sum(-1)
        routes=x[:,:,255:415].reshape(*x.shape[:2],4,5,8).transpose(2,3).flatten(3)
        encoded=self.route_embed(routes).flatten(2)
        move=self.patch_move(torch.cat((encoded,x[:,:,415:],probability.flatten(2),nonnegative,slack/20,
                                        old.masked_fill(~mask,0)[:,:,None]/8),-1))
        context=self.context(move,mask)
        phase=self.phase_next(context[:,0])
        pair=torch.cat((move,context),-1)
        risk=self.risk_head(pair).squeeze(-1)
        correction=(self.patch_scores(pair)*phase.softmax(-1)[:,None,:]).sum(-1)
        scores=old+correction+2*F.logsigmoid(risk)
        return scores.masked_fill(~mask,-1e6),grad,risk,phase

    def heads(self,x,mask):
        scores,grad,_,_=self.parts(x,mask)
        return scores,grad
