"""Estimate directional gains and losses relative to the action of the frozen policy."""
from io import BytesIO
import numpy as np
import torch
from torch import nn
from panel_policy import Panel
from protected_policy import Protected


class PairHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,20),nn.SiLU())
        self.difference=nn.Sequential(nn.Linear(60,32),nn.SiLU(),nn.Linear(32,4))

    def forward(self,x,legal):
        h=self.encoder(x)
        pooled=h.masked_fill(~legal[:,:,None],-100.).amax(1)
        u=h[:,:,None,:].expand(-1,-1,8,-1)
        v=h[:,None,:,:].expand(-1,8,-1,-1)
        shared=pooled[:,None,None,:].expand(-1,8,8,-1)
        logits=self.difference(torch.cat((u,v,shared),-1))
        reverse=logits.transpose(1,2)
        # Equal treatment of both choices under exchanging their roles.
        change=(logits[:,:,:,0:2]+reverse[:,:,:,0:2])/2
        sign=logits[:,:,:,2:4]-reverse[:,:,:,2:4]
        return change,sign


class Paired(Protected):
    def __init__(self,p,ablate=False):
        super().__init__(p,ablate)
        # The two evaluators reuse the same locked early and late weights.
        self.panel=Panel(dict(p,residual=torch.zeros(7)),ablate)
        self.heads=[]
        for state in p["pair_heads"]:
            h=PairHead(p["pair_width"]).eval().requires_grad_(False)
            h.load_state_dict(state)
            self.heads.append(h)
        self.theta=p["paired"].numpy()

    @torch.inference_mode()
    def early_parts(self,v,past):
        q,k,slack,protected=self.foundation(v,past)
        features=np.zeros((8,12))
        if protected or not 0<=slack<=6:
            return q,features
        _,x,legal,_=self.panel.features(v,past)
        if x is None:
            return q,features
        parent=int(q.argmax())
        bx=torch.from_numpy(x[None]);bm=torch.from_numpy(legal[None])
        values=[]
        for h in self.heads:
            change,sign=h(bx,bm)
            values.append((change[0,:,parent].sigmoid()*torch.tanh(sign[0,:,parent]/2)).numpy())
        arr=np.array(values)
        for j,(gamma,cut,guard) in enumerate(((0,.02,0),(0,.07,0),(0,.14,0),
                    (.12,.04,0),(.30,.08,0),(.30,.16,0),(0,.04,1),(.16,.09,1),
                    (.30,.12,2),(.08,.04,2),(0,.07,3),(.25,.10,3))):
            if guard==1 and slack>2 or guard==2 and slack>1:
                continue
            if guard==3 and (max(len(v.memory),len(v.memory[0]))<31 or min(len(v.memory),len(v.memory[0]))<27):
                continue
            per=arr[:,:,0]+gamma*arr[:,:,1]
            gain=per.mean(0)-per.std(0)
            best=int(np.where(legal,gain,-3).argmax())
            if gain[best]>cut:
                features[best,j]=q[parent]-q[best]+1
        return q,features

    def early_scores(self,v,past):
        if not self.theta.any():
            return self.foundation(v,past)[0]
        q,fields=self.early_parts(v,past)
        return q+fields@self.theta


def load(blob,ablate=False):
    pack=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"]!="paired-viability-1":
        raise ValueError("Unknown policy format")
    return Paired(pack,ablate)
