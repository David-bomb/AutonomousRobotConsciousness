"""A trainable one-split route between two frozen action heads, using visible fixed landmarks."""
from functools import lru_cache
from io import BytesIO
import numpy as np
import torch
from proper_policy import Policy as Proper
from service_window_policy import Mixture


@lru_cache(maxsize=256)
def layout_features(w,h,goal,fuel,repair):
    sep=lambda p,q:max(abs(p[0]-q[0]),abs(p[1]-q[1]))
    to_b=[sep(goal,p) for p in fuel+repair]
    between=[sep(f,r) for f in fuel for r in repair]
    return (min(w,h),max(w,h),min(to_b),max(to_b),min(between),max(between))


@lru_cache(maxsize=256)
def sites(memory):
    return tuple(tuple((x,y) for y,row in enumerate(memory) for x,c in enumerate(row) if c==symbol)
                 for symbol in "FR")


def select(rule,features):
    if rule["feature"]<0:return rule["left"]
    return rule["left"] if features[rule["feature"]]<=rule["threshold"] else rule["right"]


class Conditional(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        b=BytesIO();torch.save(pack["early"],b)
        self.proper=Proper(b.getvalue())
        self.setting=pack["proper"].tolist();self.rule=pack["rule"]

    def choose(self,v,cache=None):
        if v.last_move is None:self.previous=(None,None)
        old=(None,None) if self.ablate else self.previous;k=v,old
        if cache is None or k not in cache:
            x,y=v.position;fuel,repair=sites(v.memory)
            features=layout_features(len(v.memory[0]),len(v.memory),(x+v.dx,y+v.dy),fuel,repair)
            if select(self.rule,features):
                a=self.proper.options(v,old,[self.setting],self.proper.masks(v))[0]
            else:
                q=self.late_scores(v) if self.route(v,self.cfg) else self.early_scores(v,old)
                a=int(q.argmax())
            m=self.early.MOVES[a]
            if cache is not None:cache[k]=m
        else:m=cache[k]
        self.previous=(v.fuel_distance,v.repair_distance)
        return m


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),map_location="cpu",weights_only=True)
    if pack["format"]!="layout-conditional-1":raise ValueError("Unexpected format")
    torch.set_num_threads(1)
    return Conditional(pack,ablate)
