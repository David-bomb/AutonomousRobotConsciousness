"""Frozen service-direction classifier and terminal-reward action-score coefficients."""
from io import BytesIO
from importlib import import_module
import numpy as np
import torch
from geodesic_probe import Evidence


class Early:
    def __init__(self,base_bytes,likelihood):
        self.api=import_module("results.E-0016.prev-two.adapter")
        self.net=self.api.from_bytes(base_bytes).net
        self.geometry=import_module("results.E-0016.policy").geometry
        self.evidence=Evidence()
        self.likelihood=np.asarray(likelihood)
        self.previous=(None,None)

    @torch.inference_mode()
    def scores(self,v,old):
        pack=self.net.features(v,old)
        q,signs,_=self.net.heads(*self.api.tensor_batch([pack]))
        q=q[0].numpy();legal=pack[-1]
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        fields=np.zeros((8,8))
        if legal.sum()==1:return q,fields
        distances=(v.fuel_distance,v.repair_distance)
        stocks=(v.fuel,v.health)
        ks=[k for k in (0,1) if distances[k]>0]
        if not ks:return q,fields
        k=min(ks,key=lambda j:stocks[j]-distances[j]);slack=stocks[k]-distances[k]
        if slack>12:return q,fields
        x,y=v.position
        _,routes=self.geometry(v.memory,v.position,(x+v.dx,y+v.dy))
        required=routes[legal,1].min()>stocks[k]
        core=pack[0][:,1:41].reshape(8,5,8)
        if (core[legal,:,6]>0).any():return q,fields
        obs=self.evidence.features(v,old)[:,k]
        idx=1-k if v.health<v.fuel else k
        sensor=signs[0,:,idx,:].softmax(-1)[:,0].numpy()
        xc=np.column_stack((obs,sensor,np.clip(q,-3,0)))
        pred=xc@self.likelihood
        ok=legal&(obs[:,1]>0)
        if not ok.any():return q,fields
        belief=np.zeros(8)
        belief[ok]=np.exp(pred[ok]-pred[ok].max())
        proof=obs[:,0]
        fields[:,0]=belief*(required and slack<=1)
        fields[:,1]=belief*(required and 1<slack<=5)
        fields[:,2]=belief*(required and 5<slack<=12)
        fields[:,3]=belief*(not required and slack<=5)
        fields[:,4]=proof*(required and slack<=5)
        fields[:,5]=proof*(required and 5<slack)
        fields[:,6]=ok*(required and slack<=0)
        fields[:,7]=ok*(required and 0<slack<=2)
        return q,fields

    def choose(self,v):
        if v.last_move is None:self.previous=(None,None)
        prior=(None,None) if self.ablate else self.previous
        q,f=self.scores(v,prior)
        self.previous=(v.fuel_distance,v.repair_distance)
        return self.api.MOVES[int((q+f@self.mix).argmax())]


def load(checkpoint,ablate=False):
    pack=torch.load(BytesIO(checkpoint),map_location="cpu",weights_only=True)
    if pack["format"]!="first-unknown-edge-1":raise ValueError("Unknown format")
    base=BytesIO();torch.save(pack["base"],base)
    p=Early(base.getvalue(),pack["sensor"].numpy())
    p.mix=pack["mix"].numpy();p.ablate=ablate
    return p
