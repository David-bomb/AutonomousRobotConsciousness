"""Какие первые ходы ведут на кратчайший путь при любом допустимом месте выхода."""
from functools import lru_cache

from .frontier_automaton import _paths
from .frontier_reserve import _optimistic
from .grid import destination,wave
from .interface import Move
from .passage_automaton import exits
from .street_trunk import StreetTrunk


@lru_cache(maxsize=128)
def service_field(memory,sites):
    return {p:n for p,n,_ in wave(memory,sites)}


@lru_cache(maxsize=128)
def compatible_exits(memory,position,measured,old,prior,sites):
    known=service_field(memory,sites)
    optimistic=service_field(_optimistic(memory),sites)
    opened=_paths(memory,position)
    before=_paths(memory,prior) if prior is not None and old is not None else {}
    result=[]
    for q,(dx,dy),t,first,_ in exits(memory,position):
        p=q[0]-dx,q[1]-dy
        left=measured-t
        if not optimistic.get(q,float("inf"))<=left<=known.get(q,float("inf")):
            continue
        if opened.get(q,(float("inf"),))[0]<t:
            continue
        if old is not None and min(before.get(q,(float("inf"),))[0]+left,
                                   before.get(p,(float("inf"),))[0]+1+left)<old:
            continue
        result.append((p,q,t-1,first))
    return tuple(result)


@lru_cache(maxsize=128)
def certified(memory,position,measured,old,prior,sites):
    if not measured or not sites:
        return frozenset(),frozenset()
    physical={m:q for m in Move if (q:=destination(memory,position,m)) is not None}
    known=service_field(memory,sites)
    if known.get(position)==measured:
        sure=frozenset(m for m,q in physical.items() if known.get(q)==measured-1)
        return sure,frozenset(physical)
    possible=set()
    certain=None
    for _,_,_,first in compatible_exits(memory,position,measured,old,prior,sites):
        options=set(first)
        possible.update(options)
        certain=options if certain is None else certain & options
    return frozenset(certain or ()),frozenset(possible)


class ProvenApproach(StreetTrunk):
    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        sure,possible=certified(view.memory,view.position,readings[kind],old[kind],prior,sites[kind])
        retained={m:v for m,v in options.items() if m in (sure or possible)}
        return super().rank_directions(view,readings,old,prior,sites,retained or options,kind,default)


def smallest_detour(options, guaranteed, possible, reference=None):
    """Выбрать единственную ближайшую к показанию оценку длины пути."""
    admissible = guaranteed or possible
    lengths = {move: length for move, length in options.items() if move in admissible} or options
    if len(lengths) < 2:
        return None
    # Показание одно для всех ходов. Наибольшая нижняя оценка требует
    # наименьшего дополнительного обхода в ещё не обследованной области.
    if reference is None:
        differences = {move: -length for move, length in lengths.items()}
    else:
        measured, other_measured, other_lower = reference
        differences = {move: abs(length * other_measured - measured * other_lower)
                       for move, length in lengths.items()}
    closest = min(differences.values())
    best = [move for move, difference in differences.items() if difference == closest]
    return best[0] if len(best) == 1 else None


class SmallestDetour(StreetTrunk):
    def reference(self, view, readings, sites, kind):
        return None

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        measured = readings[kind]
        stock = (view.fuel, view.health)[kind]
        if measured is not None and 0 <= stock - measured < 2:
            guaranteed, possible = certified(view.memory, view.position, measured, old[kind], prior, sites[kind])
            move = smallest_detour(options, guaranteed, possible, self.reference(view, readings, sites, kind))
            if move is not None:
                return move
        return super().rank_directions(view, readings, old, prior, sites, options, kind, default)


class GaugeDetour(SmallestDetour):
    """Оценивает удлинение скрытых дорог по фактическому показанию до другого сервиса."""
    def reference(self, view, readings, sites, kind):
        other = 1 - kind
        if readings[other] and sites[other]:
            lower = service_field(_optimistic(view.memory), sites[other]).get(view.position)
            if lower:
                return readings[kind], readings[other], lower
        return None
