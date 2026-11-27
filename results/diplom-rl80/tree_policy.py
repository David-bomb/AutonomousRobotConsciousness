"""Soft geodesic estimates assuming a possible tree, evaluated on the visible snapshot."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from renewable_policy import targets
from service_window_policy import Mixture


class TreeEvidence:
    def __init__(self):
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.dll=ctypes.CDLL(str(Path(__file__).with_name("tree_paths.dll")))
        self.dll.tree_evidence.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,np.ctypeslib.ndpointer(np.float64)]
        self.dll.tree_evidence.restype=None

    def features(self,v,past):
        w=len(v.memory[0]);x,y=v.position;p=y*w+x;old=p
        if v.moved:
            dx,dy=v.last_move.value;old-=dy*w+dx
        now=np.array([v.fuel_distance,v.repair_distance],np.int32)
        before=np.array([-1 if z is None else z for z in past],np.int32)
        out=np.empty((8,2,6))
        self.dll.tree_evidence("".join(v.memory).encode("ascii"),w,len(v.memory),p,old,
                         targets(v.memory,(x+v.dx,y+v.dy)),now,before,out)
        return out


class TreeNet(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.tree=TreeEvidence()
        self.sensor=pack["sensor"].numpy()
        self.residual=pack["residual"].numpy()

    @torch.inference_mode()
    def early_parts(self,v,past):
        pack=self.a.features(v,past);legal=pack[-1]
        q,sign,_=self.a.heads(*self.early.tensor_batch([pack]));q=q[0].numpy()
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        fields=np.zeros((8,7));ds=(v.fuel_distance,v.repair_distance);stocks=(v.fuel,v.health)
        active=[k for k in (0,1) if ds[k]>0]
        if not active or legal.sum()<2:return q,fields
        k=min(active,key=lambda j:stocks[j]-ds[j]);slack=stocks[k]-ds[k]
        if slack>10 or ds[k]>55:return q,fields
        values=self.tree.features(v,past)[:,k]
        if not values[:,1].any():return q,fields
        idx=1-k if v.health<v.fuel else k
        sensor=sign[0,:,idx,:].softmax(-1)[:,0].numpy()
        x=np.column_stack((values,sensor,np.clip(q,-3,0)))
        allowed=legal & pack[3][:,idx,0]
        if not allowed.any():return q,fields
        loglik=x@self.sensor
        probs=np.zeros(8);probs[allowed]=np.exp(loglik[allowed]-loglik[allowed].max())
        fields[:,0]=probs*(slack<=2)
        fields[:,1]=probs*(slack>2)
        fields[:,2]=values[:,1]
        fields[:,3]=values[:,2]
        fields[:,4]=values[:,0]
        fields[:,5]=(values[:,1]>0)*(slack<=2)
        fields[:,6]=sensor
        return q,fields

    def early_scores(self,v,past):
        q,x=self.early_parts(v,past)
        return q+x@self.residual


def load(blob,ablate=False):
    pack=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"]!="tree-length-window-1":raise ValueError("Unknown model format")
    torch.set_num_threads(1)
    return TreeNet(pack,ablate)
