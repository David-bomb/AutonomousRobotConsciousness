"""Learn the value of the next service observation before committing to a branch."""
import ctypes
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from protected_policy import Protected
from renewable_policy import targets
from origin_trace import origins
from service_window_policy import coverage


class Future:
    def __init__(self,samples):
        self.samples=samples
        self.lib=ctypes.CDLL(str(Path(__file__).with_name("sensor_lookahead.dll")))
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.lib.sensor_lookahead.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ints,ints,ints,ci,ci,ints,ci,ci,ci,
                                            np.ctypeslib.ndpointer(np.float64)]
        self.lib.sensor_lookahead.restype=None

    def estimate(self,v,past):
        w=len(v.memory[0]);x,y=v.position;pos=y*w+x;prior=pos
        if v.moved:
            dx,dy=v.last_move.value;prior-=dx+w*dy
        starts=origins(v) if max(coverage(v.memory))<9 else ()
        start_array=np.asarray(starts,dtype=np.int32).reshape(-1,3)
        ds=np.array([v.fuel_distance,v.repair_distance],np.int32);old=np.array([-1 if n is None else n for n in past],np.int32)
        vals=np.empty((8,16))
        self.lib.sensor_lookahead("".join(v.memory).encode("ascii"),w,len(v.memory),pos,prior,
            targets(v.memory,(x+v.dx,y+v.dy)),ds,old,v.fuel,v.health,start_array,len(start_array),int(v.last_move is None),
            self.samples,vals)
        assert np.isfinite(vals).all() and vals[:,12:].min()>-1.e-8
        return vals


class Lookahead(Protected):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate);self.future=Future(int(pack["lookahead_samples"]))
        self.weights=pack["lookahead"].numpy()

    def early_parts(self,v,past):
        q,k,slack,protected=self.foundation(v,past);fields=np.zeros((8,10))
        if protected or not 0<=slack<=8:return q,fields
        pack=self.a.features(v,past);legal=pack[-1]
        idx=1-k if v.health<v.fuel else k
        if (legal&pack[3][:,idx,0]).sum()<2:return q,fields
        z=self.future.estimate(v,past)
        if not z.any():return q,fields
        a=int(q.argmax())
        for j,i in enumerate((0,2,4,5,6,7,12,14)):
            fields[:,j]=z[:,i]-z[a,i]
        for j,i in enumerate((4,6),8):
            best=int(np.where(legal,z[:,i],-2).argmax())
            if z[best,i]-z[a,i]>.18:
                fields[best,j]=q[a]-q[best]+1.
        return q,fields

    def early_scores(self,v,past):
        if not self.weights.any():return self.foundation(v,past)[0]
        q,x=self.early_parts(v,past)
        return q+x@self.weights


def load(blob,ablate=False):
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="partial-two-step-1":raise ValueError("Unknown policy format")
    return Lookahead(p,ablate)
