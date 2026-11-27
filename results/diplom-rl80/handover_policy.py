"""One frozen two-reading policy with a terminal-trained early action head."""
import ctypes
from importlib import import_module
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from renewable_policy import targets
from service_window_policy import Mixture,coverage


class Handover(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.sensor=pack["sensor"].numpy()
        self.residual=pack["residual"].numpy()
        self.geometry=import_module("results.E-0016.policy").geometry
        self.lib=ctypes.CDLL(str(Path(__file__).with_name("geodesic_evidence.dll")))
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.lib.evidence.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,np.ctypeslib.ndpointer(np.float64)]
        self.lib.evidence.restype=None

    @torch.inference_mode()
    def early_parts(self,v,prior):
        pack=self.a.features(v,prior)
        legal=pack[-1]
        parts,signs,_=self.a.parts(*self.early.tensor_batch([pack]))
        q=(parts@self.a.mix)[0].numpy()
        q[~legal]=-1.e6
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        result=np.zeros((8,10))
        if legal.sum()==1:return q,result
        parts=parts[0].numpy()
        result[:,0]=np.tanh(parts[:,1]/4.)
        result[:,1]=1/(1+np.exp(-parts[:,7]))
        d=np.array([v.fuel_distance,v.repair_distance],np.int32)
        stock=[v.fuel,v.health]
        active=[k for k in (0,1) if d[k]>0]
        if not active:return q,result
        k=min(active,key=lambda i:stock[i]-d[i]);slack=stock[k]-d[k]
        idx=1-k if v.health<v.fuel else k
        probability=signs[0,:,idx,:].softmax(-1).numpy()
        result[:,2]=(probability[:,0]-probability[:,2])*(slack<=2)
        result[:,3]=(probability[:,0]-probability[:,2])*(2<slack<=8)
        if slack>12:return q,result
        x,y=v.position;w=len(v.memory[0]);original=y*w+x
        if v.moved:
            dx,dy=v.last_move.value;original-=dy*w+dx
        goal=(x+v.dx,y+v.dy)
        ids=targets(v.memory,goal)
        before=np.array([-1 if z is None else z for z in prior],np.int32)
        evidence=np.empty((8,2,9))
        self.lib.evidence("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,original,ids,d,before,evidence)
        for j in range(2,8):evidence[:,:,j]/=np.maximum(evidence[:,:,j].sum(0),1.e-7)
        observed=evidence[:,k]
        allowed=legal&(observed[:,1]>0)
        if not allowed.any():return q,result
        raw=np.column_stack((observed,probability[:,0],np.clip(q,-3,0)))@self.sensor
        likelihood=np.zeros(8)
        likelihood[allowed]=np.exp(raw[allowed]-raw[allowed].max())
        _,paths=self.geometry(v.memory,v.position,goal)
        required=paths[legal,1].min()>stock[k]
        result[:,4]=likelihood*(slack<=2)
        result[:,5]=likelihood*(2<slack<=7)
        result[:,6]=likelihood*(7<slack<=12)
        result[:,7]=observed[:,0]*(slack<=6)
        result[:,8]=allowed*(required and slack<=1)
        result[:,9]=observed[:,8]*(slack>0)
        return q,result

    def early_scores(self,v,prior):
        q,parts=self.early_parts(v,prior)
        return q+parts@self.residual


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="trained-handover-1":raise ValueError("Unexpected model format")
    torch.set_num_threads(1)
    return Handover(pack,ablate)
