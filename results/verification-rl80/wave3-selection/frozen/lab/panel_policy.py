"""Fuse independent trained spatial and sensor estimates at ambiguous service forks."""
from importlib import import_module
from io import BytesIO
import numpy as np
import torch
from geodesic_probe import Evidence
from service_window_policy import Mixture,coverage
from tree_policy import TreeEvidence


class Panel(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.variants=[]
        for state in pack["variants"]:
            alt=self.early.PrevNet("plain").eval().requires_grad_(False)
            # Storing only the changed tensors also proves exactly which features are shared.
            alt.load_state_dict(dict(pack["early"],**state))
            self.variants.append(alt)
        self.geo=Evidence();self.tree=TreeEvidence()
        field=import_module("results.E-0016.sequence.spatial.field_model")
        self.field=field.FieldNet().eval().requires_grad_(False);self.field.load_state_dict(pack["field"])
        self.map_image=import_module("results.E-0016.sequence.spatial.model").image
        loc=import_module("results.E-0016.sequence.spatial.local_model")
        self.local=loc.LocalNet().eval().requires_grad_(False);self.local.load_state_dict(pack["local"])
        self.patch=loc.local_image
        self.mix=pack.get("classifier")
        self.residual=pack.get("residual",torch.zeros(7)).numpy()

    @torch.inference_mode()
    def features(self,v,past,collect=False):
        pack=self.a.features(v,past);legal=pack[-1];arg=self.early.tensor_batch([pack])
        logits,sign,risk=self.a.heads(*arg)
        def norm(z):
            z=z.copy();z[~legal]=-1.e6
            return (z-z[legal].max())/max(.5,float(z[legal].std()))
        q=norm(logits[0].numpy())
        ds=(v.fuel_distance,v.repair_distance);stock=(v.fuel,v.health)
        active=[k for k in (0,1) if ds[k]>0]
        if not active or legal.sum()<2:return q,None,legal,None
        k=min(active,key=lambda j:stock[j]-ds[j]);slack=stock[k]-ds[k]
        idx=1-k if v.health<v.fuel else k
        poss=legal & pack[3][:,idx,0]
        if slack>9 or poss.sum()<2:return q,None,legal,None
        def prob(s):return s[0,:,idx,:].softmax(-1)[:,0].numpy()
        p=prob(sign);values=[np.clip(q,-5,0),p,risk[0].sigmoid().numpy()]
        preferences=[q,p]
        for net in self.variants:
            raw,sign,risk=net.heads(*arg)
            alt=norm(raw[0].numpy());ps=prob(sign)
            values.extend((np.clip(alt,-5,0),ps,risk[0].sigmoid().numpy()))
            preferences.extend((alt,ps))
        stocks=torch.tensor([[v.fuel,v.health]])
        last=torch.tensor([self.early.MOVES.index(v.last_move) if v.last_move is not None else 8])
        board,meta=self.map_image(v)
        sg,_,valid=self.field(torch.from_numpy(board[None]),torch.from_numpy(meta[None]),stocks,last)
        assert np.array_equal(valid[0].numpy(),legal)
        spatial=sg[0].sigmoid().numpy()
        patch,meta2=self.patch(v)
        sg,quality,valid=self.local(torch.from_numpy(patch[None]),torch.from_numpy(meta2[None]),stocks,last)
        assert np.array_equal(valid[0].numpy(),legal)
        local=sg[0,:,k,:].softmax(-1).numpy();quality=quality[0].sigmoid().numpy()
        values.extend((spatial[:,k],spatial[:,3],spatial[:,4],local[:,0],local[:,2],quality[:,0],quality[:,1]))
        preferences.extend((spatial[:,k],local[:,0]))
        late=norm(self.late_scores(v));values.append(np.clip(late,-5,0));preferences.append(late)
        tree=self.tree.features(v,past)[:,k];geo=self.geo.features(v,past)[:,k]
        values.extend(tree.T);values.extend(geo.T)
        if tree[:,1].any():preferences.append(tree[:,1])
        vals=np.array(preferences)
        maxim=np.where(legal,vals,-1.e6).max(1,keepdims=True)
        votes=np.isclose(vals,maxim,atol=1.e-6).mean(0)
        values.extend((votes,poss,stock[k]/100.+np.zeros(8),np.full(8,slack/10),
                       np.full(8,ds[k]/50),np.full(8,max(coverage(v.memory))==9),np.full(8,legal.sum()/8),
                       np.full(8,0 if past[k] is None else ds[k]-past[k]),np.full(8,float(past[k] is None))))
        x=np.column_stack(values).astype("f4");x[~legal]=0
        return q,x,legal,(k,slack,geo[:,1]>0,votes)

    def early_parts(self,v,past):
        q,x,legal,info=self.features(v,past)
        out=np.zeros((8,7))
        if x is None:return q,out
        k,slack,mask,votes=info
        logits=x@self.mix["w"].numpy()+self.mix["bias"].numpy()
        proba=1/(1+np.exp(-np.clip(logits,-40,40)))
        out[:,0]=proba[:,0];out[:,1]=proba[:,1]
        out[:,2:4]=proba*(slack<=2)
        out[:,4]=votes*(slack<=3);out[:,5]=mask*(slack<=1)
        out[:,6]=np.clip(q,-3,0)*((votes.max()-votes)<.06)
        return q,out

    def early_scores(self,v,past):
        q,x=self.early_parts(v,past)
        return q+x@self.residual


def load(raw,ablate=False):
    p=torch.load(BytesIO(raw),weights_only=True,map_location="cpu")
    if p["format"]!="learned-service-panel-1":raise ValueError("Unknown format")
    return Panel(p,ablate)
