"""После разделения подходов станции могут соединяться отдельной дорогой."""
from collections import defaultdict
from functools import lru_cache
from math import isfinite

import numpy as np

from .conditional_mission import possible_worlds
from .leg_survival import local_roads
from .standby_turn import RankedStandby
from .street_trunk import contains_cycle


@lru_cache(maxsize=96)
def link_chances(view, gauges, old, prior, sites):
    stocks = view.fuel,view.health
    if (not all(sites) or any(d is None or not 0<d<=s for d,s in zip(gauges,stocks))
            or min(s-d for s,d in zip(stocks,gauges))>=2):
        return ()
    budget = min(99,max(view.health-gauges[0],view.fuel-gauges[1]))
    if sum(gauges)<=budget or budget<1:
        return ()
    x,y = view.position
    w,h = len(view.memory[0]),len(view.memory)
    origin = y*w+x
    markers,worlds = possible_worlds(view.memory,(x+view.dx,y+view.dy))
    groups = [[markers.index((k,b*w+a)) for a,b in group] for k,group in zip("FR",sites)]
    total = np.zeros(min(gauges))
    connected = total.copy()
    for opened,fields,_ in worlds:
        fs,rs = (fields[g].min(axis=0) for g in groups)
        lengths = fs[origin],rs[origin]
        if not all(isfinite(d) for d in lengths):
            continue
        error = sum(abs(d-a) for d,a in zip(lengths,gauges))
        delta = sum(abs(f[prior[1]*w+prior[0]]-d-(o-a)) for f,d,o,a in zip((fs,rs),lengths,old,gauges)
                    if prior is not None and o is not None)
        mass = 1/((1+error)*(1+delta))**2
        slopes = local_roads(opened,w,h)
        front = {origin:mass}
        for t in range(min(len(total),int(min(lengths)))):
            following = defaultdict(float)
            for p,weight in front.items():
                fuel = [q for q in slopes(p) if fs[q] == fs[p]-1]
                repair = [q for q in slopes(p) if rs[q] == rs[p]-1]
                share = weight/(len(fuel)*len(repair))
                for a in fuel:
                    for b in repair:
                        if a==b:
                            following[a] += share
                            continue
                        total[t] += share
                        # Станции должны действительно достигаться своего вида
                        # убывающим ходом, а связь должна обходить место разделения.
                        nearest = (fields[j,markers[i][1]] for i in groups[0] if fields[i,a]==fs[a]
                                   for j in groups[1] if fields[j,b]==rs[b])
                        length = min(nearest)
                        if length < fs[p]+rs[p] and length <= budget:
                            connected[t] += share
            front = following
            if not front:
                break
    return tuple(connected/(1+total))


class LinkedServices(RankedStandby):
    cross_links = staticmethod(link_chances)


class SeenLinkedServices(LinkedServices):
    @staticmethod
    def cross_links(view, readings, old, prior, sites):
        return link_chances(view,readings,old,prior,sites) if contains_cycle(view.memory,view.position) else ()
