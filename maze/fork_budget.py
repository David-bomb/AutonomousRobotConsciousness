"""Первый ход оценивается по запасу на ошибки в ещё не обследованных развилках."""
from functools import lru_cache
from math import comb, isfinite

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .grid import cell,destination,locations
from .interface import Move
from .pilot_automaton import Mode,separation
from .resource_automaton import service_order
from .road_hypotheses import RangeRoads,networks


@lru_cache(maxsize=4096)
def chance(k1,k2,room1,room2):
    """На двустороннем неизвестном выборе ошибка требует пробы и возврата."""
    if min(room1,room2)<0:
        return 0.
    a,b = room1//2,room2//2
    if a>=k1 and b>=k1+k2:
        return 1.
    ways = sum(comb(k1,i)*sum(comb(k2,j) for j in range(min(k2,b-i)+1))
               for i in range(min(k1,a,b)+1))
    return ways/2**(k1+k2)


def fork_counts(pred,target,uncertain):
    counts = {target:0}
    def lookup(p):
        trail = []
        while p not in counts:
            trail.append(p)
            p = int(pred[p])
            if p<0:
                return len(trail)
        length = counts[p]
        for node in reversed(trail):
            length += int(uncertain[node])
            counts[node] = length
        return length
    return lookup


@lru_cache(maxsize=96)
def scores(view,old):
    grid = view.memory
    h,w = len(grid),len(grid[0])
    index = lambda p:p[1]*w+p[0]
    center = index(view.position)
    readings = view.fuel_distance,view.repair_distance
    sites = [(i,p) for i,k in enumerate("FR") for p in locations(grid,k)]
    ids = [index(p) for _,p in sites]
    kinds = np.array([k for k,_ in sites])
    incomplete = np.array([any(cell(grid,x+dx,y+dy)=="?" for dx,dy in (m.value for m in Move))
                           for y in range(h) for x in range(w)])
    prior = None
    if view.moved:
        dx,dy = view.last_move.value
        prior = index((view.position[0]-dx,view.position[1]-dy))
    choices = {}
    for m in Move:
        q = destination(grid,view.position,m)
        if q is not None:
            k = cell(grid,*q)
            choices[m] = index(q),(100 if k=="F" else view.fuel-1,100 if k=="R" else view.health-1)
    vote = dict.fromkeys(choices,0.)
    for graph in networks(grid):
        fields,pred = dijkstra(graph,directed=False,unweighted=True,indices=ids,return_predecessors=True)
        ds = fields[:,center]
        near = [min(ds[kinds==i]) for i in range(2)]
        if not all(map(isfinite,near)):
            continue
        error = sum(abs(n-r) for n,r in zip(near,readings))
        gradient = 0.
        if prior is not None:
            gradient = sum(abs(min(fields[kinds==i,prior])-near[i]-(old[i]-readings[i]))
                           for i in range(2) if old[i] is not None)
        weight = 1/((1+error)*(1+gradient))**2
        undirected = (graph+graph.T).tocsr()
        forks = incomplete & (np.diff(undirected.indptr)>2)
        counts = [fork_counts(p,t,forks) for p,t in zip(pred,ids)]
        best = dict.fromkeys(choices,0.)
        for a,(i,pa) in enumerate(sites):
            for b,(j,pb) in enumerate(sites):
                if i==j or not isfinite(ds[a]+ds[b]):
                    continue
                leg_a = readings[i]+ds[a]-near[i]
                leg_b = readings[j]+ds[b]-near[j]
                common = (ds[a]+ds[b]-fields[a,ids[b]])/(2*min(ds[a],ds[b])) if min(ds[a],ds[b]) else 0.
                between = max(separation(pa,pb),round(leg_a+leg_b-2*min(leg_a,leg_b)*common))
                if 2*between>100:
                    continue
                k2 = counts[b](ids[a])
                for m,(q,stocks) in choices.items():
                    if min(stocks)<=0:
                        continue
                    d1 = 0 if q==ids[a] else max(1,round(readings[i]+fields[a,q]-near[i]))
                    if stocks[i]<d1 or stocks[j]<=d1:
                        continue
                    prob = chance(counts[a](q),k2,stocks[i]-d1,stocks[j]-d1-between)
                    best[m] = max(best[m],prob)
        for m in vote:
            vote[m] += weight*best[m]
    return vote


class ForkBudget(RangeRoads):
    def _fallback_move(self,view,goal,stops):
        old = tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        demand = service_order(view,self._fallback.mode,self.RESERVE)
        default = super()._fallback_move(view,goal,stops)
        if demand is None or None in (view.fuel_distance,view.repair_distance):
            return default
        legal = [m for m in Move if destination(view.memory,view.position,m) is not None]
        if len(legal)<2:
            return default
        current = view.fuel_distance,view.repair_distance
        if view.moved and all(p is not None for p in old):
            dx,dy = view.last_move.value
            reverse = Move((-dx,-dy))
            if all(a>b for a,b in zip(current,old)):
                return reverse
            if len(legal)==2 and all(a<b for a,b in zip(current,old)):
                return next(m for m in legal if m!=reverse)
        vote = scores(view,old)
        move = max(vote,key=lambda m:(vote[m],m==default))
        if move!=default:
            self._fallback.returning = False
            self.activity = "Запас на неизвестные развилки"
        return move
