"""Одна сеть с замороженным основанием mission600 и двумя скалярами состояния."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch import nn
from maze.grid import destination
from maze.interface import Move,Observation
from ..policy import geometry
from ..sequence.model import SequenceNet,encode

MOVES=tuple(Move)
BY_VECTOR={m.value:m for m in MOVES}
CORE=244
PER=16


def memory_fields(view,prev,legal):
    x,y=view.position
    _,raw=geometry(view.memory,view.position,(x+view.dx,y+view.dy))
    # Длины от текущей клетки с заданным первым ходом. Возврат через
    # текущую клетку выделен отдельно через фактическое показание D.
    lengths=raw[:,:15].reshape(8,5,3)[:,1:].reshape(8,2,2,3).min(2)
    curr=np.array([view.fuel_distance,view.repair_distance])
    stocks=np.array([view.fuel,view.health])
    assert all(z is not None for z in curr)
    curr=curr.astype(int)
    prev=np.array([z if z is not None else -1 for z in prev])
    known=prev>=0
    shift=curr-prev
    near=curr-stocks
    reverse=np.zeros(8,bool)
    hop=np.full(8,2)
    if view.last_move is not None:
        dx,dy=view.last_move.value if view.moved else (0,0)
        prior=(x-dx,y-dy)
        for a,m in enumerate(MOVES):
            q=(x+m.value[0],y+m.value[1])
            reverse[a]=q==prior
            if q==prior: hop[a]=0
            elif (edge:=BY_VECTOR.get((q[0]-prior[0],q[1]-prior[1]))) is not None:
                if destination(view.memory,prior,edge)==q: hop[a]=1
    lower=np.maximum(0,np.maximum(curr-1,np.minimum(curr+1,lengths[:,:,1]-1)))
    upper=np.minimum(curr+1,lengths[:,:,0]-1)
    lower=np.maximum(lower,np.where(known,prev-hop[:,None],-1000))
    upper=np.minimum(upper,np.where(known,prev+hop[:,None],1000))
    signs=np.array([-1,0,1])
    possible=(curr[None,:,None]+signs>=lower[:,:,None])&(curr[None,:,None]+signs<=upper[:,:,None])
    for k in (0,1):
        decreasing=legal&possible[:,k,0]
        if curr[k]>0 and decreasing.sum()==1:
            possible[decreasing,k,1:]=False
    assert possible[legal].any(2).all()
    fields=np.stack(np.broadcast_arrays(
        known, np.where(known,np.minimum(prev,100)/50,0),
        np.where(known,shift,0),known&(shift>0),known&(shift<0),known&(shift==0),
        (reverse[:,None]&known), hop[:,None]*known/2,
        (lower-curr).clip(-1,1),(upper-curr).clip(-1,1),
        (stocks/100),curr/50,np.clip(-near/10,-3,3),near>=-3,
        np.clip((lengths[:,:,1]-curr)/5,-3,3),lengths[:,:,0]==curr),axis=-1).astype(np.float32)
    if view.health<view.fuel:
        fields=fields[:,::-1].copy()
        possible=possible[:,::-1].copy()
    fields[~legal]=0
    possible[~legal]=True
    return fields,possible


class PrevNet(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.base=SequenceNet(kind)
        self.base.requires_grad_(False)
        self.sign=nn.Sequential(nn.Linear(PER+3,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,3))
        self.edge=nn.Sequential(nn.Linear(CORE+2*(PER+3),96),nn.LayerNorm(96),nn.SiLU(),
                                nn.Linear(96,64),nn.SiLU())
        self.action=nn.Sequential(nn.Linear(128,48),nn.SiLU(),nn.Linear(48,1))
        self.risk=nn.Linear(64,1)
        self.gains=nn.Linear(12,1,bias=False)
        self.mix=nn.Parameter(torch.tensor([1.,1.,0.,0.,0.,0.,0.,0.,0.]))
        nn.init.zeros_(self.sign[-1].weight); nn.init.zeros_(self.sign[-1].bias)
        nn.init.zeros_(self.action[-1].weight); nn.init.zeros_(self.action[-1].bias)
        nn.init.zeros_(self.gains.weight)

    def compressed(self,view):
        prior=self.base.wall_prior.detach().cpu().numpy().tobytes()
        x,ok=encode(view,prior)
        with torch.no_grad():
            q,g=self.base.heads(torch.from_numpy(x[None]),torch.from_numpy(ok[None]))
        logits=q[0].numpy()
        old=(logits-logits[ok].max()).clip(-40,0)
        grad=g[0].numpy()
        grad-=grad.max(2,keepdims=True)
        # Геометрия всей миссии, характеристики целей и открываемых окон из исходного кодирования.
        site=x[:,:110].reshape(8,5,22)
        context=np.concatenate((x[:,255:],site[:,:,[0,1,2,3,4,6,7]].reshape(8,-1),x[:,110:120],
                                np.tile([view.dx/50,view.dy/50],(8,1))),axis=1)
        v=np.column_stack((old,context)).astype(np.float32)
        assert v.shape==(8,CORE)
        v[~ok]=0
        return v,grad,ok

    def features(self,view,previous):
        core,grad,legal=self.compressed(view)
        past,possible=memory_fields(view,previous,legal)
        return core,grad,past,possible,legal

    def parts(self,core,grad,past,possible,legal):
        signs=(grad+self.sign(torch.cat((past,grad/6),-1))).masked_fill(~possible,-1000)
        prob=signs.softmax(-1)
        sensor=torch.cat((past,prob),-1).flatten(2)
        scaled=torch.cat((core[:,:,:1]/8,core[:,:,1:]),-1)
        h=self.edge(torch.cat((scaled,sensor),-1))
        ctx=h.masked_fill(~legal[:,:,None],-1e5).amax(1,keepdim=True).expand_as(h)
        # Все коэффициенты полезности уменьшения/роста измерения обучаемы.
        urgency=past[:,:,:,13]
        evidence=torch.stack((prob[:,:,:,0],prob[:,:,:,2],prob[:,:,:,0]*urgency,
                    prob[:,:,:,2]*urgency,prob[:,:,:,0]*urgency*past[:,:,:,3],
                    prob[:,:,:,2]*urgency*past[:,:,:,3]),-1).flatten(2)
        delta=self.action(torch.cat((h,ctx),-1)).squeeze(-1)+self.gains(evidence).squeeze(-1)
        # Базис последнего линейного слоя. Ни одна компонента не ограничивает легальные ходы.
        slack=past[:,:,:,12]
        priority=slack.masked_fill(past[:,:,:,11]<=0,10).argmin(2,keepdim=True)
        take=lambda arr:arr.gather(2,priority).squeeze(2)
        d=take(prob[:,:,:,0]-prob[:,:,:,2])
        tight=take(urgency); margin=take(slack)
        active=(take(past[:,:,:,11])>0).float()
        probe=take(past[:,:,:,3])*tight
        known=core[:,:,1:41].reshape(*core.shape[:2],5,8)
        feasible=(known[:,:,:,6]>.5)&(known[:,:,:,2]>=0)
        lengths=known[:,:,:,4].masked_fill(~feasible,8).amin(2)
        shortest=lengths.masked_fill(~legal,8).amin(1,keepdim=True)
        target=-(75*(lengths-shortest)).clamp(0,8)/2*(shortest<8)
        frontier=core[:,:,1+160:1+172]
        valid_front=core[:,:,1+184:1+196]>.5
        fdist=(frontier[:,:,5]*50).masked_fill(~valid_front[:,:,5],300.)
        fbest=fdist.masked_fill(~legal,300.).amin(1,keepdim=True)
        survey=-(fdist-fbest).clamp(0,16)/4*(fbest<300)*(shortest>=8)
        risk=self.risk(h).squeeze(-1)
        parts=torch.stack((core[:,:,0],delta,d*tight*active,d*((margin>.3)&(margin<=1.2))*active,
                           d*probe*active,target,survey,risk.clamp(-6,6),d*(margin<=0)*active),-1)
        return parts,signs,risk

    def heads(self,core,grad,past,possible,legal):
        parts,signs,risk=self.parts(core,grad,past,possible,legal)
        return (parts@self.mix).masked_fill(~legal,-1e6),signs,risk

    def forward(self,*args):
        return self.heads(*args)[0]


def tensor_batch(pack):
    return tuple(torch.from_numpy(np.stack(group)).to(torch.float32 if i<3 else torch.bool)
                 for i,group in enumerate(zip(*pack)))


class Controller:
    def __init__(self,path,expected_sha256,ablate=False):
        data=Path(path).read_bytes()
        if sha256(data).hexdigest()!=expected_sha256: raise ValueError("Не совпал SHA-256 весов")
        self._load(data,ablate)

    def _load(self,data,ablate):
        state=torch.load(BytesIO(data),map_location="cpu",weights_only=True)
        self.net=PrevNet("rich" if "base.base.wall_prior" in state else
                         "critical" if "base.base.risk.0.weight" in state else "plain")
        self.net.load_state_dict(state)
        self.net.requires_grad_(False).eval()
        self.ablate=ablate
        self.reset()

    def reset(self):
        self.previous=(None,None)

    @torch.inference_mode()
    def choose(self,view:Observation,cache=None)->Move:
        if view.last_move is None: self.reset()
        previous=self.previous
        key=(view,previous)
        if cache is not None and key in cache:
            move=cache[key]
        else:
            values=self.net.features(view,(None,None) if self.ablate else previous)
            scores=self.net(*tensor_batch([values]))
            move=MOVES[int(scores.argmax())]
            if cache is not None: cache[key]=move
        self.previous=(view.fuel_distance,view.repair_distance)
        return move


def from_bytes(data,ablate=False):
    obj=Controller.__new__(Controller)
    obj._load(data,ablate)
    return obj
