"""Learned eight-action scores using the ordinary Observation and fixed train parameters."""
import ctypes
from functools import lru_cache
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3] / "University/Diplom"
sys.path.insert(0, str(PROJECT))
base = import_module("results.E-0016.policy")


class PrefixPolicy:
    def __init__(self, base_weights, prefix_bank, parameters=None):
        self.net=base.Policy().eval()
        self.net.load_state_dict(torch.load(base_weights,map_location="cpu",weights_only=True))
        self.lib=ctypes.CDLL(str(HERE/"gradient_prefix.dll"))
        self.lib.load_bank.argtypes=[ctypes.c_char_p]
        self.lib.load_bank.restype=ctypes.c_int
        self.lib.predict.argtypes=[ctypes.c_char_p]+[ctypes.c_int]*5+[np.ctypeslib.ndpointer(np.float64)]
        self.lib.predict.restype=None
        if self.lib.load_bank(os.fsencode(prefix_bank))<=0:
            raise OSError("Empty or unreadable prefix bank")
        if parameters is not None:
            self.weights=np.asarray(json.loads(Path(parameters).read_bytes())["coefficients"])

    @lru_cache(maxsize=4096)
    def prefix(self, memory, pos, kind, distance):
        w,h=len(memory[0]),len(memory)
        counts=np.zeros(8)
        self.lib.predict("".join(memory).encode("ascii"),w,h,pos[1]*w+pos[0],distance,ord(kind),counts)
        maximum=counts.max()
        return counts / maximum if maximum else counts

    def features(self, view):
        x,legal=base.features(view)
        with torch.inference_mode():
            logits=self.net(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))[0].numpy()
        pos=view.position
        goal=(pos[0]+view.dx,pos[1]+view.dy)
        _,raw=base.geometry(view.memory,pos,goal)
        paths=raw[:,:15].reshape(8,5,3)
        stocks=[view.fuel,view.health]
        sensor=[view.fuel_distance,view.repair_distance]
        urgent=sorted(range(2),key=lambda i:stocks[i]-sensor[i])
        idx,other=urgent
        slack=stocks[idx]-sensor[idx]

        def reading(i):
            if sensor[i]==0 or stocks[i]-sensor[i]>35:
                return np.zeros(8)
            return self.prefix(view.memory,pos,"FR"[i],sensor[i])

        mass=reading(idx)
        both=reading(other) if stocks[other]-sensor[other]<=10 else np.zeros(8)
        first=1+idx*2
        confirmed=paths[:,first:first+2,0].min(1)
        optimistic=paths[:,first:first+2,1].min(1)
        margin=min(stocks)
        finish=paths[:,0,0]
        goal_lower=paths[:,0,1]
        enough=slack>8 and margin>=12
        frontier=raw[:,15]
        prior=np.asarray(view.last_move.value if view.last_move is not None else (0,0))
        reverse=(base.XY==-prior).all(1)
        fields=np.stack([
            mass*(slack<=2),mass*(2<slack<=8),mass*(8<slack<=20),mass*(20<slack),
            both, ((optimistic<=sensor[idx]) & (slack<=6)),
            ((confirmed==sensor[idx]) & (slack<=12)),
            reverse*(slack<=3),
            ((finish==finish[legal].min()) & (finish<=margin)),
            np.exp(-np.minimum(frontier,80)/8)*enough,
            -np.minimum(goal_lower-goal_lower[legal].min(),20)/10*enough,
            (frontier<=2)*enough
        ],axis=1).astype(np.float32)
        fields[~legal]=0
        return logits,fields

    def choose(self, view):
        baseline,fields=self.features(view)
        return base.MOVES[int((baseline+fields@self.weights).argmax())]
