"""Обучаемое пространственное поле возможных путей и оценка каждого физического действия."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import SIDE,MOVES,convolution


class FieldNet(nn.Module):
    def __init__(self):
        super().__init__()
        y,x=torch.meshgrid(torch.arange(SIDE),torch.arange(SIDE),indexing="ij")
        self.register_buffer("xy",torch.stack((x,y),-1).float())
        self.register_buffer("shift",torch.tensor([m.value for m in MOVES],dtype=torch.float32))
        self.local=convolution(14,16)
        self.mid=nn.Sequential(convolution(16,24,2),convolution(24,24))
        self.wide=nn.Sequential(convolution(24,32,2),convolution(32,32,dilation=2),convolution(32,32,dilation=3))
        self.merge=convolution(32+24,24)
        self.decode=nn.Sequential(convolution(24+16,24),convolution(24,24))
        self.field=nn.Conv2d(24,4,1)
        self.direction=nn.Sequential(nn.Linear(24*3+32+4+2+4+2,80),nn.LayerNorm(80),nn.SiLU(),nn.Linear(80,48),nn.SiLU())
        self.q=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,5))

    def forward(self,image,meta,stocks,last):
        b=len(meta)
        points=torch.stack((meta[:,:6]%SIDE,meta[:,:6]//SIDE),-1).float()
        xy=self.xy[None,None]-points[:,:,None,None,:]
        man=xy.abs().sum(-1);cheb=xy.abs().amax(-1)
        marker=(man==0).float()
        ground,owner=image[:,0],image[:,1]
        d=meta[:,6:].float()
        folds=[]
        for k,s in ((1,0),(3,1)):
            for metric in (man,cheb):
                diff=d[:,s,None,None]-metric[:,0]-metric[:,k:k+2].amin(1)
                folds.append((diff/15).clamp(-2,2))
        planes=torch.stack(((ground==1),(ground==2),owner>0,marker[:,0],marker[:,1:3].amax(1),marker[:,3:5].amax(1),
                 marker[:,5],xy[:,0,:,:,0]/30,xy[:,0,:,:,1]/30,cheb[:,5]/30,*folds),1).float()
        a=self.local(planes);m=self.mid(a);wide=self.wide(m)
        joined=self.merge(torch.cat((F.interpolate(wide,scale_factor=2,mode="nearest"),m),1))
        cells=self.decode(torch.cat((F.interpolate(joined,scale_factor=2,mode="nearest"),a),1))
        field=self.field(cells)
        bits=((owner[:,None].int()>>torch.arange(8,device=image.device)[None,:,None,None])&1).float().flatten(2)
        region=torch.bmm(bits,cells.flatten(2).transpose(1,2))/bits.sum(2,keepdim=True).clamp_min(1)
        pos=points[:,0].long()
        next_pos=(pos[:,None]+self.shift[None]).clamp(0,SIDE-1).long()
        batch=torch.arange(b,device=image.device)[:,None]
        xs,ys=next_pos[:,:,0],next_pos[:,:,1]
        near=cells.permute(0,2,3,1)[batch,ys,xs]
        source=cells.permute(0,2,3,1)[batch,pos[:,None,1],pos[:,None,0]].expand(-1,8,-1)
        local_score=field.permute(0,2,3,1)[batch,ys,xs]
        global_state=wide.mean((2,3))[:,None,:].expand(-1,8,-1)
        rotated=torch.arange(8,device=image.device)[None,:]
        inverse=torch.tensor([2,3,0,1,6,7,4,5,8],device=image.device)
        previous=torch.stack((last[:,None]==rotated,inverse[last.long()][:,None]==rotated),-1).float()
        resources=torch.cat((stocks.float()/100,(stocks.float()-d).clamp(-20,35)/20),1)[:,None].expand(-1,8,-1)
        vec=self.direction(torch.cat((near,source,region,global_state,local_score,previous,resources,
                                      self.shift[None].expand(b,-1,-1)),-1))
        legal=(((owner[batch,ys,xs].int()>>rotated)&1)>0)
        context=vec.masked_fill(~legal[:,:,None],-1e5).amax(1,keepdim=True).expand_as(vec)
        value=self.q(torch.cat((vec,context),-1))
        # Три первых оценки — первый шаг кратчайшего пути к F, R, B.
        return torch.cat((value[:,:,:3]+local_score[:,:,:3],value[:,:,3:]),-1),field,legal
