"""Фиксированная передача между обученными сетями по раскрытым окнам сервисов."""
from functools import lru_cache
from io import BytesIO
import torch
from .adapter import PrevNet,MOVES,tensor_batch
from ..sequence.belief.joint_model import Joint
import numpy as np


@lru_cache(maxsize=512)
def complete(memory):
    kinds=set()
    for y,row in enumerate(memory):
        for x,symbol in enumerate(row):
            if symbol in "FR" and symbol not in kinds:
                window=[r[max(0,x-1):x+2] for r in memory[max(0,y-1):y+2]]
                if sum(len(r)-r.count("?") for r in window)==9:kinds.add(symbol)
    return len(kinds)==2


class Controller:
    def __init__(self,pack,ablate):
        if pack["format"]!="prev-two-early-joint-v1":raise ValueError("Неизвестный комплект")
        state=pack["early"]
        kind="rich" if "base.base.wall_prior" in state else "critical" if "base.base.risk.0.weight" in state else "plain"
        self.early=PrevNet(kind)
        self.early.load_state_dict(state);self.early.eval().requires_grad_(False)
        self.late=Joint(pack["late"])
        self.linear=pack["late"]["linear"].numpy()
        self.ablate=ablate
        self.reset()

    def reset(self):
        self.previous=(None,None)

    @torch.inference_mode()
    def choose(self,v,cache=None):
        if v.last_move is None:self.reset()
        key=v,self.previous
        if cache is not None and key in cache: move=cache[key]
        else:
            if complete(v.memory):
                features,legal=self.late.directions(v)
                action=int(np.where(legal,features@self.linear,-1e12).argmax())
            else:
                previous=(None,None) if self.ablate else self.previous
                pack=self.early.features(v,previous)
                action=int(self.early(*tensor_batch([pack])).argmax())
            move=MOVES[action]
            if cache is not None:cache[key]=move
        self.previous=v.fuel_distance,v.repair_distance
        return move


def load(data,ablate=False):
    return Controller(torch.load(BytesIO(data),weights_only=True,map_location="cpu"),ablate)
