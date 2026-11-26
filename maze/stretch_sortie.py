"""Выбор выезда по допустимому удлинению неизвестных дорог до финиша."""
from collections import deque
from functools import lru_cache
from math import ceil, inf

from .excursion_automaton import lookouts
from .frontier_automaton import _arrive, _paths, _service_states
from .frontier_reserve import _optimistic
from .grid import cell, destination
from .interface import Move
from .pilot_automaton import Mode
from .street_trunk import StreetTrunk


@lru_cache(maxsize=96)
def outside_legs(memory, goal, stops, root):
    inside = _paths(memory, root)
    if goal in inside:
        return (), {}, ()
    markers = [(q,k) for q,k in stops if q not in inside] + [(goal,"B")]
    outside = _optimistic(memory)
    gateways = set(lookouts(memory, goal)).intersection(inside)
    from_exit = {p:[] for p in gateways}
    between = [[] for _ in markers]
    for target,_ in markers:
        queue = deque([target])
        distances = {target:0}
        exits = {}
        while queue:
            p = queue.popleft()
            length = distances[p]+1
            for move in Move:
                q = destination(outside, p, move)
                if q is None or q in distances:
                    continue
                if q in inside:
                    if q in gateways:
                        exits.setdefault(q,length)
                else:
                    distances[q] = length
                    queue.append(q)
        for p in gateways:
            from_exit[p].append(exits.get(p,inf))
        for row,(p,_) in zip(between,markers):
            row.append(distances.get(p,inf))
    return tuple(k for _,k in markers),from_exit,tuple(tuple(row) for row in between)


def tolerates(stocks, first, between, kinds, scale):
    queue=[(-1,*stocks)]
    seen={}
    while queue:
        src,f,h=queue.pop()
        prior=seen.setdefault(src,[])
        if any(a>=f and b>=h for a,b in prior):
            continue
        prior.append((f,h))
        legs=first if src==-1 else between[src]
        for j,(kind,length) in enumerate(zip(kinds,legs)):
            if j==src or length*scale>min(f,h):
                continue
            remaining=_arrive(kind,ceil(length*scale),f,h)
            if remaining:
                if kind=="B":
                    return True
                queue.append((j,*remaining))
    return False


@lru_cache(maxsize=4096)
def tolerance(stocks, first, between, kinds):
    if not tolerates(stocks,first,between,kinds,1.):
        return 0.
    low,high=1.,101.
    # Верхняя граница — запас на один ход. Остановиться, когда неопределённость
    # по коэффициенту меньше одной сотой длины хода.
    while high-low>=0.01:
        mid=(low+high)/2
        if tolerates(stocks,first,between,kinds,mid):
            low=mid
        else:
            high=mid
    return low


class StretchSortie(StreetTrunk):
    def _reconsider_sortie(self,view,goal,stops,planned):
        memory=view.memory
        kinds,exits,between=outside_legs(memory,goal,stops,min(_paths(memory,view.position)))
        if not exits:
            return None
        best=None
        for traveled,f,h,first,roads in _service_states(memory,view.position,view.fuel,view.health,stops):
            for p,legs in exits.items():
                if p==view.position or p not in roads:
                    continue
                length,move=roads[p]
                after=_arrive(cell(memory,*p),length,f,h) if length else (f,h)
                if after is None:
                    continue
                scale=tolerance(after,tuple(legs),between,kinds)
                if scale:
                    rank=(scale,-traveled-length,-legs[-1],p)
                    if best is None or rank>best[0]:
                        best=rank,first or move
        if best:
            self._fallback.mode=Mode.GOAL
            self._fallback.returning=False
            self._fallback.last_lengths={Mode.FUEL:view.fuel_distance,Mode.REPAIR:view.repair_distance}
            self.activity="Выезд с запасом для удлинения дороги"
            return best[1]
        return None
