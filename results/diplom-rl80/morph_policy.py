"""Use fully compatible room/extra-opening hypotheses for ambiguous early choices."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from origin_trace import origins
from renewable_policy import targets
from service_window_policy import Mixture,coverage


class Terrain:
    def __init__(self,samples):
        self.samples=samples;ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32);floats=np.ctypeslib.ndpointer(np.float64)
        self.dll=ctypes.CDLL(str(Path(__file__).with_name("morphology_worlds.dll")))
        self.dll.morph_votes.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,ci,ints,ci,ci,ci,floats,floats]
        self.dll.morph_votes.restype=None

    def get(self,v,past):
        w=len(v.memory[0]);x,y=v.position;pos=y*w+x;oldp=pos
        if v.moved:
            dx,dy=v.last_move.value;oldp-=dx+w*dy
        starts=origins(v) if max(coverage(v.memory))<9 else ()
        where=np.asarray(starts,dtype=np.int32).reshape(-1,3)
        ds=np.array([v.fuel_distance,v.repair_distance],np.int32);old=np.array([-1 if n is None else n for n in past],np.int32)
        out=np.empty((6,8,3));stats=np.empty((4,3))
        self.dll.morph_votes("".join(v.memory).encode("ascii"),w,len(v.memory),pos,oldp,targets(v.memory,(x+v.dx,y+v.dy)),
                             ds,old,v.fuel,v.health,where,len(where),int(v.last_move is None),self.samples,out,stats)
        assert np.isfinite(out).all()
        return out,stats


class MorphNet(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.terrain=Terrain(int(pack["samples"]))
        self.coeff=pack["coeff"].numpy()

    @torch.inference_mode()
    def early_parts(self,v,past):
        p=self.a.features(v,past);legal=p[-1]
        q,sign,_=self.a.heads(*self.early.tensor_batch([p]));q=q[0].numpy()
        q=(q-q[legal].max())/max(.5,float(q[legal].std()));fields=np.zeros((8,10))
        ds=(v.fuel_distance,v.repair_distance);s=(v.fuel,v.health)
        good=[k for k in (0,1) if ds[k]>0]
        if legal.sum()<2 or not good:return q,fields
        k=min(good,key=lambda j:s[j]-ds[j]);slack=s[k]-ds[k]
        idx=1-k if v.health<v.fuel else k;possible=legal&p[3][:,idx,0]
        if slack>7 or possible.sum()<2:return q,fields
        vals,n=self.terrain.get(v,past)
        if not n[:,0].any():return q,fields
        fields[:,:3]=vals[:3,:,k].T
        fields[:,3]=vals[1,:,k]*(slack<=2)
        # Greater posterior support for a family increases its contribution.
        support=(n[:,1]+4*n[:,2])/(1+n[:,0])
        fields[:,4]=vals[3,:,k]*max(support[:2])
        fields[:,5]=vals[4,:,k]*support[2]
        fields[:,6]=vals[5,:,k]*support[3]
        fields[:,7]=vals[1,:,2]
        fields[:,8]=sign[0,:,idx,:].softmax(-1)[:,0].numpy()
        fields[:,9]=possible*(slack<=1)
        return q,fields

    def early_scores(self,v,past):
        q,x=self.early_parts(v,past)
        return q+x@self.coeff


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="conditional-morphology-1":raise ValueError("Unrecognized format")
    return MorphNet(p,ablate)
