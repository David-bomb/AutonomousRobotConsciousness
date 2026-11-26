"""Оценивать удлинение неизвестной дороги по длинам, измеренным в реальной позиции."""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .grid import locations
from .road_hypotheses import lattice
from .simple_exit_paths import simple_exit_probabilities
from .standby_turn import RankedStandby


@lru_cache(maxsize=64)
def road_edges(memory):
    w,h = len(memory[0]),len(memory)
    board = np.frombuffer("".join(memory).encode(),dtype="S1")
    edges = lattice(w,h)
    edges = edges[:,(board[edges]!=b"#").all(axis=0)]
    unknown = (board[edges]==b"?").any(axis=0)
    return np.r_[edges[0],edges[1]],np.r_[edges[1],edges[0]],np.r_[unknown,unknown]


def scaled_graph(memory,scale,sinks=()):
    a,b,hidden = road_edges(memory)
    size = len(memory)*len(memory[0])
    lengths = np.where(hidden,scale,1.)
    valid = ~np.isin(a,sinks) if len(sinks) else slice(None)
    return coo_matrix((lengths[valid],(a[valid],b[valid])),shape=(size,size)).tocsr()


@lru_cache(maxsize=96)
def measured_scale(memory,origin,readings):
    width = len(memory[0])
    groups = [(np.array([p[1]*width+p[0] for p in locations(memory,k)]),d)
              for k,d in zip("FR",readings) if d is not None and d>0]
    if not groups:
        return 1.
    def explains(scale):
        distances = dijkstra(scaled_graph(memory,scale),indices=origin[1]*width+origin[0])
        return all(min(distances[g])>=d-1.e-9 for g,d in groups)
    if explains(1.):
        return 1.
    low,high = 1.,float(max(d for _,d in groups))
    while high-low>1/64:
        mid = (low+high)/2
        if explains(mid):
            high = mid
        else:
            low = mid
    return high


@lru_cache(maxsize=96)
def first_lengths(memory,goal,points,scale):
    w = len(memory[0])
    ids = np.array([p[1]*w+p[0] for p in points],dtype=int)
    graph = scaled_graph(memory,scale,ids)
    field = dijkstra(graph,indices=goal[1]*w+goal[0],directed=True)
    return dict(zip(points,field[ids]))


class GaugeExits(RankedStandby):
    strict = False

    def sortie_curves(self,view,goal,points):
        original = simple_exit_probabilities(view.memory,goal,points)
        scale = measured_scale(view.memory,view.position,(view.fuel_distance,view.repair_distance))
        if scale==1.:
            return original
        lower = first_lengths(view.memory,goal,points,1.)
        adjusted = first_lengths(view.memory,goal,points,scale)
        stocks = np.arange(101)
        curves = {}
        for p,curve in original.items():
            a,b = lower[p],adjusted[p]
            if b<=a or not np.isfinite(b):
                curves[p] = curve
            elif self.strict:
                curves[p] = curve*(stocks>=b-1.e-9)
            else:
                curves[p] = curve*np.clip((stocks-a)/(b-a),0.,1.)
        return curves if any(c[-1] for c in curves.values()) else original


class FittedGaugeExits(GaugeExits):
    strict = True
