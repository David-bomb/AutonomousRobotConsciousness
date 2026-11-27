"""Одни параметры читают дорожное окружение в системе каждого допустимого хода."""
import numpy as np
import torch
from torch import nn
from .model import MOVES,image

R=6


def local_image(v):
    board,m=image(v)
    # Эту операцию выполняем над реально возвращённой памятью, включая область за рамкой.
    x,y=v.position
    patch=np.zeros((2,2*R+1,2*R+1),np.uint8);patch[0]=2
    xmin,xmax=max(0,x-R),min(39,x+R);ymin,ymax=max(0,y-R),min(39,y+R)
    patch[:,ymin-y+R:ymax-y+R+1,xmin-x+R:xmax-x+R+1]=board[:,ymin:ymax+1,xmin:xmax+1]
    return patch,m


class LocalNet(nn.Module):
    def __init__(self):
        super().__init__()
        transforms=[];indices=[]
        turns=np.array([[0,-1],[1,0]])
        for a in range(8):
            rotate=np.linalg.matrix_power(turns,a%4)
            symmetric=np.array([[-1,0],[0,1]]) if a<4 else np.array([[0,-1],[-1,0]])
            group=[rotate,rotate@symmetric];transforms.append(group)
            j=[]
            for mat in group:
                ids=[]
                for dy in range(-R,R+1):
                    for dx in range(-R,R+1):
                        u,w=mat@np.array([dx,dy]);ids.append((w+R)*(2*R+1)+u+R)
                j.append(ids)
            indices.append(j)
        self.register_buffer("matrices",torch.from_numpy(np.array(transforms)).float())
        self.register_buffer("index",torch.tensor(indices))
        self.register_buffer("canonical",torch.tensor([[0,-1]]*4+[[1,-1]]*4).float())
        self.register_buffer("moves",torch.tensor([m.value for m in MOVES]).long())
        self.road=nn.Sequential(nn.Conv2d(4,12,3),nn.SiLU(),nn.Conv2d(12,16,3,stride=2),nn.SiLU(),
                nn.Conv2d(16,20,3),nn.SiLU(),nn.Flatten(),nn.Linear(20*3*3,40),nn.SiLU())
        self.point=nn.Sequential(nn.Linear(12,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,24),nn.SiLU())
        self.hidden=nn.Sequential(nn.Linear(40+24*3+3,88),nn.LayerNorm(88),nn.SiLU(),nn.Linear(88,48),nn.SiLU())
        self.geodesic=nn.Sequential(nn.Linear(96,48),nn.SiLU(),nn.Linear(48,9))
        self.resource=nn.Sequential(nn.Linear(96+6+9,48),nn.SiLU(),nn.Linear(48,2))

    def forward(self,p,meta,stocks,last):
        b=len(p)
        transformed=p.flatten(2)[:,:,self.index].permute(0,2,3,1,4)
        floor=transformed[:,:,:,0];owner=transformed[:,:,:,1].int()
        bits=torch.arange(8,device=p.device)[None,:,None,None]
        channels=torch.stack((floor==1,floor==2,owner>0,((owner>>bits)&1)>0),3).float()
        shape=self.road(channels.reshape(-1,4,13,13)).reshape(b,8,2,40)
        coords=torch.stack((meta[:,:6]%40,meta[:,:6]//40),-1).float()
        delta=coords[:,1:]-coords[:,:1]
        v=torch.einsum("bsk,agkc->bagsc",delta,self.matrices)
        man=v.abs().sum(-1);cheb=v.abs().amax(-1)
        shift=self.canonical[None,:,None,None,:]
        d=meta[:,6:].float()
        distance=torch.stack((d[:,0],d[:,0],d[:,1],d[:,1],torch.zeros_like(d[:,0])),1)[:,None,None,:]
        values=[v[:,:,:,:,0]/30,v[:,:,:,:,1]/30,man/45,cheb/35,distance/45,(distance-man)/20,
                (distance-cheb)/20,man-(v-shift).abs().sum(-1),cheb-(v-shift).abs().amax(-1),
                (v[:,:,:,:,0].abs()%2),(v[:,:,:,:,1].abs()%2),(v*shift).sum(-1)/35]
        points=self.point(torch.stack(torch.broadcast_tensors(*values),-1))
        axis_parity=torch.einsum("bk,agkc->bagc",coords[:,0]%2,self.matrices.abs())
        diagonal=(self.canonical.abs().sum(1)-1)[None,:,None,None].expand(b,-1,2,-1)
        parity=torch.cat((axis_parity,diagonal),-1)
        hidden=self.hidden(torch.cat((shape,points[:,:,:,:2].amax(3),points[:,:,:,2:4].amax(3),
                                      points[:,:,:,4],parity),-1)).mean(2)
        ids=(R+self.moves[:,1])*13+R+self.moves[:,0]
        ok=((p[:,1].flatten(1)[:,ids].int()>>torch.arange(8,device=p.device)[None])&1).bool()
        context=hidden.masked_fill(~ok[:,:,None],-1e5).amax(1,keepdim=True).expand_as(hidden)
        common=torch.cat((hidden,context),-1)
        g=self.geodesic(common).reshape(b,8,3,3)
        inv=torch.tensor([2,3,0,1,6,7,4,5,8],device=p.device)
        action=torch.arange(8,device=p.device)[None]
        res=torch.cat((stocks.float()/100,(stocks.float()-d).clamp(-20,40)/20),1)[:,None,:].expand(-1,8,-1)
        extra=torch.cat((res,(last[:,None]==action).float()[:,:,None],(inv[last.long()][:,None]==action).float()[:,:,None]),-1)
        value=self.resource(torch.cat((common,extra,g.softmax(-1).flatten(2)),-1))
        return g,value,ok
