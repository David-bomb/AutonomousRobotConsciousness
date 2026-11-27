"""Learn the consequences of one early decision followed by an autonomous policy."""
from io import BytesIO
import numpy as np
import torch
from torch import nn
from geodesic_probe import Evidence
from service_window_policy import Mixture,coverage
from tree_policy import TreeEvidence


class BranchHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.read=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))

    def forward(self,x,legal):
        h=self.embed(x)
        pooled=h.masked_fill(~legal[:,:,None],-100.).amax(1,keepdim=True)
        return self.read(torch.cat((h,pooled.expand_as(h)),dim=2))


class BranchNet(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.geodesic=Evidence();self.tree=TreeEvidence()
        self.head=None
        if "head" in pack:
            self.head=BranchHead(int(pack["width"])).eval().requires_grad_(False)
            self.head.load_state_dict(pack["head"])
        self.residual=pack.get("residual",torch.zeros(7)).numpy()

    @torch.inference_mode()
    def input(self,v,past,expanded=True):
        pack=self.a.features(v,past);legal=pack[-1]
        components,sign,risk=self.a.parts(*self.early.tensor_batch([pack]))
        parts=components[0].numpy();q=(components@self.a.mix)[0].numpy()
        q[~legal]=-1.e6
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        ds=(v.fuel_distance,v.repair_distance);stocks=(v.fuel,v.health)
        active=[k for k in (0,1) if ds[k]>0]
        if legal.sum()<2 or not active:return q,None,legal,None
        k=min(active,key=lambda j:stocks[j]-ds[j]);slack=stocks[k]-ds[k]
        idx=1-k if v.health<v.fuel else k
        possible=legal & pack[3][:,idx,0]
        if slack>10 or possible.sum()<2:return q,None,legal,None
        info=k,slack
        if not expanded:return q,None,legal,info
        prob=sign[0].softmax(-1).numpy()
        geo=self.geodesic.features(v,past)[:,k]
        tree=self.tree.features(v,past)[:,k]
        x0,y0=v.position;h,w=len(v.memory),len(v.memory[0])
        fresh=[]
        for move in self.early.MOVES:
            dx,dy=move.value
            x,y=x0+dx,y0+dy
            fresh.append(sum(v.memory[j][i]=="?" for j in range(max(0,y-1),min(h,y+2))
                                                 for i in range(max(0,x-1),min(w,x+2)))/5.)
        seen=coverage(v.memory)
        available=[max(-1.,min(2.,(stocks[j]-ds[j])/8)) for j in (k,1-k)]
        shared=[*available,ds[k]/40.,ds[1-k]/60.,max(seen)/9.,min(seen)/9.,legal.sum()/8.,possible.sum()/8.,
                0. if past[k] is None else ds[k]-past[k],float(past[k] is None),abs(v.dx)/40.,abs(v.dy)/40.]
        features=np.column_stack((np.clip(q,-5,0),np.tanh(parts/5.),risk[0].sigmoid().numpy(),
               prob[:,idx,:],prob[:,1-idx,:],pack[3][:,idx,:],geo,tree,fresh,np.tile(shared,(8,1)))).astype("f4")
        features[~legal]=0.
        return q,features,legal,info

    @torch.inference_mode()
    def early_parts(self,v,past):
        q,x,legal,info=self.input(v,past)
        fields=np.zeros((8,7))
        if x is None:return q,fields
        scores=self.head(torch.from_numpy(x[None]),torch.from_numpy(legal[None])).sigmoid()[0].numpy()
        fields[:,0]=scores[:,0];fields[:,1]=scores[:,1];fields[:,2]=scores[:,2]
        fields[:,3:6]=scores*(info[1]<=2)
        baseline=q.argmax()
        fields[:,6]=q<q[baseline]-.1
        return q,fields

    def early_scores(self,v,past):
        q,f=self.early_parts(v,past)
        return q+f@self.residual


def load(blob,ablate=False):
    pack=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"]!="early-rollout-return-1":raise ValueError("Unknown policy type")
    return BranchNet(pack,ablate)
