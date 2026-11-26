"""Проверять, устойчив ли выбор пути к случайному сокращению числа вариантов дороги."""
from functools import lru_cache

from .frontier_reserve import _lengths, _optimistic
from .path_particles import path_fractions
from .standby_turn import RankedStandby


@lru_cache(maxsize=96)
def estimates(view,readings,old,prior,sites,options,kind,exact):
    if exact:
        row = path_fractions(view,readings,old,prior,sites,options,kind,limit=512,exact=True)
        return (row,) if row is not None else ()
    return tuple(path_fractions(view,readings,old,prior,sites,options,kind,limit=96,seed=i) for i in range(4))


class ExactApproaches(RankedStandby):
    exact = True

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        parent = super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        d = readings[kind]
        if d is None or not 0<=(view.fuel,view.health)[kind]-d<2 or len(options)<2:
            return parent
        lower = _lengths(_optimistic(view.memory),view.position)
        if sum(lower.get(p,float("inf"))<=d for p in sites[kind])!=1:
            return parent
        rows = estimates(view,readings,old,prior,sites,tuple(options),kind,self.exact)
        if not rows:
            return parent
        raw = self.direction_votes(view,readings,old,prior,sites)[kind]
        winners = set()
        for row in rows:
            if not any(row.values()):
                return parent
            mean = sum(row.values())/len(row)
            winners.add(max(options,key=lambda m:(raw[m]*row.get(m,mean),m==parent)))
        return next(iter(winners)) if len(winners)==1 else parent


class RepeatedApproaches(ExactApproaches):
    exact = False
