import torch
from torch import nn


class Safety(nn.Module):
    def __init__(self):
        super().__init__()
        self.site=nn.Sequential(nn.Linear(22+22,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,24),nn.SiLU())
        self.move=nn.Sequential(nn.Linear(3*24+16+10+20,72),nn.LayerNorm(72),nn.SiLU(),nn.Linear(72,48),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96,48),nn.SiLU(),nn.Linear(48,7))

    def forward(self,x,exits,walk,ok):
        shape=(*x.shape[:2],5,22)
        site=self.site(torch.cat((x[:,:,:110].reshape(shape),exits.reshape(shape)),-1))
        r=x[:,:,255:415].reshape(*x.shape[:2],4,5,8)
        h=self.move(torch.cat((site[:,:,0],site[:,:,1:3].amax(2),site[:,:,3:].amax(2),
                               walk,x[:,:,110:120],r[:,:,:,:,6].flatten(2)),-1))
        context=h.masked_fill(~ok[:,:,None],-1e5).amax(1,keepdim=True).expand_as(h)
        out=self.head(torch.cat((h,context),-1))
        return out[:,:,:6].reshape(*x.shape[:2],2,3),out[:,:,6]
