"""Score urgent forks by simulated road geometry compatible with the observed trip."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from origin_trace import origins
from renewable_policy import targets
from service_window_policy import Mixture,coverage


class OriginNet(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.residual=pack["residual"].numpy()
        self.particles=int(pack["particles"])
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.lib=ctypes.CDLL(str(Path(__file__).with_name("origin_worlds.dll")))
        self.lib.origin_votes.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,ci,ints,ci,ci,ci,
                                        np.ctypeslib.ndpointer(np.float64)]
        self.lib.origin_votes.restype=None

    def worlds(self,v,past):
        w=len(v.memory[0]);x,y=v.position;pos=w*y+x
        prior_at=pos
        if v.moved:
            dx,dy=v.last_move.value;prior_at=pos-dx-w*dy
        ids=targets(v.memory,(x+v.dx,y+v.dy))
        before=np.array([-1 if z is None else z for z in past],np.int32)
        current=np.array([v.fuel_distance,v.repair_distance],np.int32)
        starts=origins(v) if max(coverage(v.memory))<9 else ()
        rows=np.asarray(starts,dtype=np.int32).reshape(-1,3)
        vals=np.empty((3,8,4))
        self.lib.origin_votes("".join(v.memory).encode("ascii"),w,len(v.memory),pos,prior_at,ids,current,before,
                          v.fuel,v.health,rows,len(rows),int(v.last_move is None),self.particles,vals)
        assert np.isfinite(vals).all()
        return vals

    @torch.inference_mode()
    def early_parts(self,v,past):
        pack=self.a.features(v,past)
        q,sign,_=self.a.heads(*self.early.tensor_batch([pack]))
        q=q[0].numpy();legal=pack[-1]
        q=(q-q[legal].max())/max(.5,float(q[legal].std()))
        fields=np.zeros((8,8))
        ds=(v.fuel_distance,v.repair_distance);stocks=(v.fuel,v.health)
        active=[k for k in (0,1) if ds[k]>0]
        if legal.sum()<2 or not active:return q,fields
        k=min(active,key=lambda j:stocks[j]-ds[j]);slack=stocks[k]-ds[k]
        order=1-k if v.health<v.fuel else k
        possible=legal & pack[3][:,order,0]
        if slack>7 or possible.sum()<2:return q,fields
        p=self.worlds(v,past)
        sensor=sign[0,:,order,:].softmax(-1)[:,0].numpy()
        fields[:,0]=p[0,:,k]
        fields[:,1]=p[1,:,k]
        fields[:,2]=p[2,:,k]
        fields[:,3]=p[2,:,k]*(slack<=2)
        fields[:,4]=p[2,:,2]
        fields[:,5]=p[2,:,3]
        fields[:,6]=sensor
        fields[:,7]=possible*(slack<=1)
        return q,fields

    def early_scores(self,v,past):
        q,f=self.early_parts(v,past)
        return q+f@self.residual


def load(content,ablate=False):
    p=torch.load(BytesIO(content),weights_only=True,map_location="cpu")
    if p["format"]!="origin-conditioned-1":raise ValueError("Unknown frozen policy")
    torch.set_num_threads(1)
    return OriginNet(p,ablate)
