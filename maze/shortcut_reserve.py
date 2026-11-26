"""Учитывать продолжение от первой станции без возврата к общему проходу."""
from functools import lru_cache

from .frontier_automaton import _arrive
from .mission_order import first_stop_models
from .street_trunk import StreetTrunk, StreetTrunkReserve


@lru_cache(maxsize=96)
def onward_support(view,old):
    """Доля взвешенных вариантов, допускающих вторую остановку либо финиш."""
    origin=view.position
    goal=origin[0]+view.dx,origin[1]+view.dy
    readings=view.fuel_distance,view.repair_distance
    if any(d is None or d<=0 for d in readings):
        return 0.
    arrivals=[_arrive(k,d,view.fuel,view.health) for k,d in zip("FR",readings)]
    support=total=0.
    for weight,fields,sites,nearest in first_stop_models(view,goal,old):
        total+=weight
        for i,points in enumerate(nearest):
            stock=arrivals[i]
            if stock is None:
                continue
            other="RF"[i]
            onward=[j for j,(k,_) in enumerate(sites) if k==other]
            direct=any(fields[-1,p]<=min(stock) for p in points)
            reaches_station=any(_arrive(other,fields[j,p],*stock) for p in points for j in onward)
            if direct or reaches_station:
                support+=weight
                break
    return support/total if total else 0.


class ShortcutReserve(StreetTrunkReserve):
    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        args=view,readings,old,prior,sites
        restricted=super().direction_votes(*args)[kind]
        raw=StreetTrunk().direction_votes(*args)[kind]
        def best(scores):
            return max(options,key=lambda m:(scores.get(m,0.),m==default))
        first,second=best(restricted),best(raw)
        if first==second:
            return first
        p=onward_support(view,old)
        mass=sum(restricted.get(m,0.) for m in options)
        unrestricted_mass=sum(raw.get(m,0.) for m in options)
        if not mass or not unrestricted_mass:
            return first
        return best({m:(1-p)*restricted.get(m,0.)/mass+p*raw.get(m,0.)/unrestricted_mass for m in options})
