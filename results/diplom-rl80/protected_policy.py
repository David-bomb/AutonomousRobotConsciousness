"""Combine two frozen early decision heads and fit conservative latent-road corrections."""
from importlib import import_module
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import torch
from morph_policy import Terrain
from tree_policy import TreeNet


class Protected(TreeNet):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.api=import_module("results.E-0016.prev-two.cautious_policy")
        correction=self.api.Correction(int(pack["cautious_dim"])).eval().requires_grad_(False)
        correction.load_state_dict(pack["cautious_state"])
        self.cautious=SimpleNamespace(early=self.a,correction=correction)
        self.threshold=float(pack["cautious_threshold"])
        self.terrain=Terrain(int(pack["samples"]))
        self.morph=pack["morph"].numpy()

    def foundation(self,v,past):
        q,x=super().early_parts(v,past);q=q+x@self.residual
        k,slack=self.api.urgency(v);protected=False
        if 0<=slack<=2:
            _,best,strength=self.api.Controller.decision(self.cautious,v,past)
            if strength>self.threshold:
                q[best]=q.max()+1.;protected=True
        return q,k,slack,protected

    def early_parts(self,v,past):
        q,k,slack,protected=self.foundation(v,past)
        fields=np.zeros((8,12))
        if protected or not 0<=slack<=7:return q,fields
        p=self.a.features(v,past);idx=1-k if v.health<v.fuel else k
        possible=p[-1] & p[3][:,idx,0]
        if possible.sum()<2:return q,fields
        parent=int(q.argmax())
        vals,counts=self.terrain.get(v,past)
        if not counts[:,0].any():return q,fields
        switches=[(0,.29,2),(1,.29,2),(2,.29,2),(2,.44,2),(2,.59,2),(2,.75,2),
                  (2,.44,7),(2,.60,7),(4,.47,2),(5,.47,2),(3,.47,2),(1,.17,1)]
        for i,(j,minimum,max_slack) in enumerate(switches):
            if slack>max_slack:continue
            probability=vals[j,:,k];best=int(np.where(possible,probability,-2).argmax())
            if probability[best]-probability[parent]>minimum and probability[best]>=.60:
                fields[best,i]=q[parent]-q[best]+1.
        return q,fields

    def early_scores(self,v,past):
        if not self.morph.any():return self.foundation(v,past)[0]
        q,x=self.early_parts(v,past)
        return q+x@self.morph


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="protected-tree-morph-1":raise ValueError("Unknown frozen policy")
    torch.set_num_threads(1)
    return Protected(p,ablate)
