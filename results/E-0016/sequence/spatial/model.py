"""Свёрточное чтение всей памяти клеток и участков дороги за каждым первым ходом."""
import ctypes as ct
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from maze.grid import locations
from maze.interface import Move

HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/"region.dll"))
LIB.region.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.uint8)]
LIB.region.restype=None
SIDE=40
MOVES=tuple(Move)


def image(v):
    h,w=len(v.memory),len(v.memory[0])
    if max(w,h)>SIDE:raise ValueError("Уровень выходит за размер исследуемого корпуса")
    p=v.position
    z=np.empty((2,SIDE,SIDE),np.uint8)
    LIB.region("".join(v.memory).encode(),w,h,p[1]*w+p[0],z)
    points=[p,*locations(v.memory,"F"),*locations(v.memory,"R"),(p[0]+v.dx,p[1]+v.dy)]
    meta=np.array([y*SIDE+x for x,y in points]+[v.fuel_distance,v.repair_distance],np.int16)
    return z,meta


def convolution(cin,cout,stride=1,dilation=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=dilation,dilation=dilation),
                         nn.GroupNorm(4,cout),nn.SiLU())


class SpatialSensor(nn.Module):
    def __init__(self):
        super().__init__()
        y,x=torch.meshgrid(torch.arange(SIDE),torch.arange(SIDE),indexing="ij")
        self.register_buffer("xy",torch.stack((x,y),-1).float())
        self.register_buffer("shift",torch.tensor([m.value for m in MOVES],dtype=torch.float32))
        self.local=convolution(14,16)
        self.mid=nn.Sequential(convolution(16,24,2),convolution(24,24))
        self.wide=nn.Sequential(convolution(24,32,2),convolution(32,32,dilation=2),convolution(32,32,dilation=3))
        self.site=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,24),nn.SiLU())
        self.move=nn.Sequential(nn.Linear(32+32+16+16+32+24*3+2,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.out=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,9))

    def forward(self,images,meta):
        b=len(meta)
        coords=torch.stack((meta[:,:6]%SIDE,meta[:,:6]//SIDE),-1).float()
        xy=self.xy[None,None]-coords[:,:,None,None,:]
        manh=xy.abs().sum(-1)
        cheb=xy.abs().amax(-1)
        d=meta[:,6:].float()
        walls,ownership=images[:,0],images[:,1]
        onehot=(manh==0).float()
        folds=[]
        for first,meter in ((1,d[:,0]),(3,d[:,1])):
            for metric in (manh,cheb):
                diff=meter[:,None,None]-metric[:,0]-metric[:,first:first+2].amin(1)
                folds.append((diff/15).clamp(-2,2))
        planes=torch.stack(((walls==1),(walls==2),ownership>0,onehot[:,0],onehot[:,1:3].amax(1),
                            onehot[:,3:5].amax(1),onehot[:,5],xy[:,0,:,:,0]/30,xy[:,0,:,:,1]/30,
                            cheb[:,5]/30,*folds),1).float()
        one=self.local(planes)
        wide=self.wide(self.mid(one))
        region=((ownership[:,None].int()>>torch.arange(8,device=meta.device)[None,:,None,None])&1).float()
        shared=wide.mean((2,3))[:,None,:].expand(b,8,-1)
        def pool(im,mask):
            weights=mask.flatten(2)
            return torch.bmm(weights,im.flatten(2).transpose(1,2))/weights.sum(2,keepdim=True).clamp_min(1)
        patch=F.max_pool2d(region,4)
        end=(coords[:,0,None]+self.shift[None]).clamp(0,SIDE-1).long()
        batch=torch.arange(b,device=meta.device)[:,None]
        near=one.permute(0,2,3,1)[batch,end[:,:,1],end[:,:,0]]
        broad=wide.permute(0,2,3,1)[batch,end[:,:,1]//4,end[:,:,0]//4]
        dirs=self.shift[None,:,None,:]
        distance=coords[:,None,1:]-coords[:,None,0,None,:]
        before=distance.abs().sum(-1)
        after=(distance-dirs).abs().sum(-1)
        diag=distance.abs().amax(-1)
        after_diag=(distance-dirs).abs().amax(-1)
        observed=torch.stack((d[:,0],d[:,0],d[:,1],d[:,1],torch.zeros_like(d[:,0])),1)[:,None,:]
        vec=[distance[:,:,:,0]/30,distance[:,:,:,1]/30,before/40,diag/30,(observed-before)/20,
             (observed-diag)/20,observed/45,before-after,diag-after_diag,(distance*dirs).sum(-1)/30]
        sites=self.site(torch.stack(torch.broadcast_tensors(*vec),-1))
        m=self.move(torch.cat((shared,pool(wide,patch),pool(one,region),near,broad,
                  sites[:,:,:2].amax(2),sites[:,:,2:4].amax(2),sites[:,:,4],self.shift[None].expand(b,-1,-1)),-1))
        # Только реально видимые соседние дорожные клетки с разрешённым первым ребром.
        bits=ownership[batch,end[:,:,1],end[:,:,0]].int()
        legal=((bits>>torch.arange(8,device=meta.device)[None,:])&1).bool()
        context=m.masked_fill(~legal[:,:,None],-1e4).amax(1,keepdim=True).expand_as(m)
        return self.out(torch.cat((m,context),-1)).reshape(b,8,3,3),legal
