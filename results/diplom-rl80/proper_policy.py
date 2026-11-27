"""Learned selection of shortest known observation routes and a frozen early neural policy."""
import ctypes
from importlib import import_module
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from renewable_policy import targets


class Policy:
    def __init__(self,model_bytes):
        self.api=import_module("results.E-0016.prev-two.adapter")
        self.net=self.api.from_bytes(model_bytes).net
        self.lib=ctypes.CDLL(str(Path(__file__).with_name("proper_frontier.dll")))
        ci=ctypes.c_int
        ints=np.ctypeslib.ndpointer(np.int32)
        self.lib.proper.argtypes=[ctypes.c_char_p,ci,ci,ci,ints,ci,ci,ints]
        self.lib.proper.restype=None
        self.previous=(None,None)

    def masks(self,v):
        w=len(v.memory[0]);x,y=v.position
        out=np.zeros(7,np.int32)
        self.lib.proper("".join(v.memory).encode("ascii"),w,len(v.memory),w*y+x,
                        targets(v.memory,(x+v.dx,y+v.dy)),v.fuel,v.health,out)
        return out

    def options(self,v,old,configs,fields):
        masks=[]
        for safe,risk,reserve in configs:
            if not safe: masks.append(0);continue
            required=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
            risky=fields[risk] if risk and required>=reserve else 0
            masks.append(int(fields[0] or fields[safe] or risky))
        scores=None
        actions=[]
        for mask in masks:
            if mask and not mask&(mask-1):
                actions.append(mask.bit_length()-1)
            else:
                if scores is None:
                    with torch.inference_mode():
                        values=self.net.features(v,old)
                        scores=self.net(*self.api.tensor_batch([values]))[0].numpy()
                q=scores if not mask else np.where([(mask>>i)&1 for i in range(8)],scores,-np.inf)
                actions.append(int(q.argmax()))
        return actions

    def choose(self,v):
        if v.last_move is None:self.previous=(None,None)
        old=(None,None) if self.ablate else self.previous
        a=self.options(v,old,[self.configuration],self.masks(v))[0]
        self.previous=(v.fuel_distance,v.repair_distance)
        return self.api.MOVES[a]


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="proper-frontier-1":raise ValueError("Unknown format")
    core=BytesIO();torch.save(pack["base"],core)
    p=Policy(core.getvalue())
    p.configuration=pack["configuration"].tolist()
    p.ablate=ablate
    return p
