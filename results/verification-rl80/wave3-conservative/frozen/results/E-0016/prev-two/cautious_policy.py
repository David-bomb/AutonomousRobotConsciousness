"""Строгая ранняя поправка. Рабочий путь получает только Observation и F/R предыдущего хода."""
import ctypes as ct
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .adapter import MOVES,tensor_batch
from .early_policy import Controller as Pair,complete

LIB=ct.CDLL(str(Path(__file__).with_name("cautious_geometry.dll")))
INTS=np.ctypeslib.ndpointer(np.int32)
LIB.evidence.argtypes=[ct.c_char_p,*([ct.c_int]*4),INTS,INTS,INTS,np.ctypeslib.ndpointer(np.float64)]
LIB.evidence.restype=None


@lru_cache(maxsize=512)
def sites(memory,goal):
    w=len(memory[0]);ids=[goal[1]*w+goal[0]]
    for ch in "FR":ids.extend(y*w+x for y,row in enumerate(memory) for x,c in enumerate(row) if c==ch)
    assert len(ids)==5
    return np.array(ids,np.int32)


def evidence(v,previous):
    x,y=v.position;w=len(v.memory[0]);old=y*w+x
    if v.moved:
        dx,dy=v.last_move.value;old-=dy*w+dx
    d=np.array((v.fuel_distance,v.repair_distance),np.int32)
    prior=np.array([-1 if z is None else z for z in previous],np.int32)
    out=np.zeros((8,2,16))
    LIB.evidence("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,old,
                 sites(v.memory,(x+v.dx,y+v.dy)),d,prior,out)
    return out.astype(np.float32)


def urgency(v):
    d=(v.fuel_distance,v.repair_distance);stock=(v.fuel,v.health)
    k=min(range(2),key=lambda i:stock[i]-d[i] if d[i]>0 else 1000)
    return k,stock[k]-d[k]


@torch.inference_mode()
def base_parts(net,v,previous):
    pack=net.features(v,previous);parts,signs,_=net.parts(*tensor_batch([pack]))
    q=(parts@net.mix)[0].numpy().copy();q[~pack[-1]]=-1e6
    signs=signs[0].softmax(-1).numpy()
    if v.health<v.fuel:signs=signs[:,::-1]
    return q,pack[-1],signs,parts[0].numpy()


def design(v,previous,raw,base):
    q,legal,signs,parts=base;k,slack=urgency(v)
    own=int(q.argmax());order=[k,1-k];x,y=v.position
    possible=(raw[:,k,1]>.5)&legal
    gate=0<=slack<=2 and legal.sum()>1 and (possible.sum()>=2 or not possible[own])
    # Внутри входа срочный сервис первый, пары показаний связываются до перестановки.
    norm=raw.copy()
    for j in (2,3,4,5,6,7,9,10,11,12,13,14):
        norm[:,:,j]/=np.maximum(raw[:,:,j].sum(0),1e-7)
    arr=[norm[:,order].reshape(8,-1),signs[:,order].reshape(8,-1),
         np.tanh(parts[:,[0,1,2,4,7]]/5),
         np.clip((q-q[own])/max(.7,float(q[legal].std())),-6,0)[:,None]]
    d=(v.fuel_distance,v.repair_distance);stock=(v.fuel,v.health)
    traits=[]
    for a,m in enumerate(MOVES):
        dx,dy=m.value
        previous_dir=v.last_move.value if v.moved else (0,0)
        row=[float(a==own),slack/2,legal.sum()/8,possible.sum()/8,
             (dx*previous_dir[0]+dy*previous_dir[1])/2,float(m.value==previous_dir),abs(dx*dy)]
        for t in order:
            shift=0 if previous[t] is None else d[t]-previous[t]
            row.extend((d[t]/60,np.clip((stock[t]-d[t])/12,-2,2),float(previous[t] is not None),
                        shift,float(shift==-1),float(shift==1)))
        for advance,side in ((1,0),(2,0),(3,0),(2,1),(2,-1),(1,1),(1,-1),(3,1),(3,-1)):
            xx=x+advance*dx-side*dy;yy=y+advance*dy+side*dx
            cell=v.memory[yy][xx] if 0<=yy<len(v.memory) and 0<=xx<len(v.memory[0]) else "#"
            row.extend((cell=="?",cell=="#"))
        traits.append(row)
    arr.append(np.asarray(traits,np.float32))
    result=np.concatenate(arr,axis=1).astype(np.float32);result[~legal]=0
    return result,gate


class Correction(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,40),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(120,48),nn.SiLU(),nn.Linear(48,8))

    def forward(self,x,legal,own):
        h=self.edge(x)
        old=h[torch.arange(len(h)),own][:,None,:].expand_as(h)
        ctx=h.masked_fill(~legal[:,:,None],-1e4).amax(1,keepdim=True).expand_as(h)
        return self.head(torch.cat((h,ctx,old),-1))


class Controller(Pair):
    def __init__(self,pack,ablate):
        super().__init__(dict(format="prev-two-early-joint-v1",early=pack["early"],late=pack["late"]),ablate)
        self.correction=Correction(pack["dim"]);self.correction.load_state_dict(pack["correction"])
        self.correction.requires_grad_(False).eval();self.threshold=float(pack["threshold"])

    @torch.inference_mode()
    def decision(self,v,previous):
        base=base_parts(self.early,v,previous);q,legal,_,_=base;own=int(q.argmax())
        k,slack=urgency(v)
        if v.fuel_distance is None or legal.sum()<2 or slack<0 or slack>2:return own,own,0.
        e=evidence(v,previous);x,gate=design(v,previous,e,base)
        if not gate:return own,own,0.
        z=self.correction(torch.from_numpy(x[None]),torch.from_numpy(legal[None]),torch.tensor([own]))[0]
        # Величина вмешательства зависит от оценённой жизнеспособности относительно исходного хода.
        probability=z[:,0].sigmoid()
        sign=z[:,2:5].softmax(1)[:,0]
        weight=(probability+.12*sign+.16*z[:,1].sigmoid()).numpy()
        weight[~legal]=-5
        best=int(weight.argmax())
        return own,best,float(weight[best]-weight[own])

    @torch.inference_mode()
    def choose(self,v,cache=None):
        if v.last_move is None:self.reset()
        pair=self.previous;key=v,pair
        if cache is not None and key in cache:move=cache[key]
        elif complete(v.memory):
            feat,legal=self.late.directions(v)
            move=MOVES[int(np.where(legal,feat@self.linear,-1e12).argmax())]
        else:
            own,best,delta=self.decision(v,(None,None) if self.ablate else pair)
            move=MOVES[best if delta>self.threshold else own]
        if cache is not None:cache[key]=move
        self.previous=v.fuel_distance,v.repair_distance
        return move


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="cautious-prev-two-v1":raise ValueError("Неизвестный формат")
    return Controller(pack,ablate)
