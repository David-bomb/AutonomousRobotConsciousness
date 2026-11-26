"""Оценить длину общей части дорог к заправке и мастерской по текущим наблюдениям."""
from collections import defaultdict
from functools import lru_cache
from math import isfinite

import numpy as np

from .conditional_mission import possible_worlds
from .leg_survival import local_roads
from .second_window import ReturnWindows
from .shared_trunk import split_frequency


@lru_cache(maxsize=96)
def common_profile(view, readings, old, prior, sites):
    if (any(d is None or d <= 0 for d in readings) or not all(sites)
            or min(s-d for s,d in zip((view.fuel,view.health),readings)) >= 2):
        return None
    width, height = len(view.memory[0]), len(view.memory)
    def index(p):
        return p[1]*width + p[0]
    start = index(view.position)
    goal = view.position[0]+view.dx, view.position[1]+view.dy
    markers, worlds = possible_worlds(view.memory, goal)
    groups = [[markers.index((kind,index(p))) for p in group] for kind,group in zip("FR",sites)]
    n = min(readings)
    exposed = np.zeros(n)
    split = np.zeros(n)
    for opened, fields, _ in worlds:
        fuel, repair = (fields[g].min(axis=0) for g in groups)
        simulated = fuel[start],repair[start]
        if not all(isfinite(d) for d in simulated):
            continue
        error = sum(abs(a-b) for a,b in zip(simulated,readings))
        delta = sum(abs(field[index(prior)]-d - (before-actual))
                    for field,d,before,actual in zip((fuel,repair),simulated,old,readings)
                    if prior is not None and before is not None)
        weight = 1/((1+error)*(1+delta))**2
        neighbors = local_roads(opened,width,height)
        front = {start:weight}
        for t in range(min(n, int(min(simulated)))):
            onward = defaultdict(float)
            for p,mass in front.items():
                roads = neighbors(p)
                a = {q for q in roads if fuel[q] == fuel[p]-1}
                b = {q for q in roads if repair[q] == repair[p]-1}
                joint = a & b
                share = mass/(len(a)*len(b))
                exposed[t] += mass
                split[t] += mass - len(joint)*share
                for q in joint:
                    onward[q] += share
            if not onward:
                break
            front = onward
    if not exposed[0]:
        return None
    prior_split = split_frequency(view.memory)
    return tuple((split + prior_split)/(exposed + 1))


def first_split(view, readings, old, prior, sites):
    profile = common_profile(view, readings, old, prior, sites)
    return profile[:1] if profile else None


class ModeledSplit(ReturnWindows):
    split_profile = staticmethod(first_split)


class ModeledTrunk(ModeledSplit):
    split_profile = staticmethod(common_profile)
