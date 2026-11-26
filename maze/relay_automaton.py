"""Выбор первого заезда по тому, хватит ли второго ресурса на следующую дорогу."""
from functools import lru_cache
from math import isfinite

from scipy.sparse.csgraph import dijkstra

from .grid import destination,locations
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions,service_order
from .road_hypotheses import RangeRoads,networks


@lru_cache(maxsize=128)
def relay_votes(view, previous):
    width = len(view.memory[0])
    index = lambda p:p[1]*width+p[0]
    origin = index(view.position)
    neighbors = [(m,index(q)) for m in Move if (q:=destination(view.memory,view.position,m))]
    stocks = view.fuel,view.health
    actual = view.fuel_distance,view.repair_distance
    groups = [tuple(locations(view.memory,k)) for k in "FR"]
    sites = [(i,index(p)) for i,group in enumerate(groups) for p in group]
    modes = Mode.FUEL,Mode.REPAIR
    admissible = [possible_directions(view,Service(m,stocks[i],actual[i],groups[i]),previous[i])
                  for i,m in enumerate(modes)]
    prior = None
    if view.moved:
        dx,dy = view.last_move.value
        prior = index((view.position[0]-dx,view.position[1]-dy))
    votes = {}
    for graph in networks(view.memory):
        fields = dijkstra(graph,directed=False,unweighted=True,indices=[p for _,p in sites])
        lengths = [field[origin] for field in fields]
        nearest = [min(d for (k,_),d in zip(sites,lengths) if k==i) for i in range(2)]
        if not all(map(isfinite,nearest)):
            continue
        error = sum(abs(a-b) for a,b in zip(actual,nearest))
        if prior is not None:
            for i,old in enumerate(previous):
                if old is not None:
                    at_prior = min(f[prior] for (k,_),f in zip(sites,fields) if k==i)
                    error += abs((at_prior-nearest[i])-(old-actual[i]))
        best,choices = None,set()
        for a,(kind,_) in enumerate(sites):
            da = lengths[a]
            if not isfinite(da) or da==0:
                continue
            leg = actual[kind]+da-nearest[kind]
            if leg > stocks[kind] or leg >= stocks[1-kind] or 100-stocks[kind] <= leg:
                continue
            first = [m for m,j in neighbors if fields[a,j]==da-1 and (
                da!=nearest[kind] or m in admissible[kind])]
            if not first:
                continue
            for b,(other,site) in enumerate(sites):
                db = lengths[b]
                if other==kind or not isfinite(db) or db==0:
                    continue
                second = actual[other]+db-nearest[other]
                # Сегмент, общий двум дорогам из текущей клетки, проходится один раз.
                shared = (da+db-fields[a,site])/(2*min(da,db))
                between = round(leg+second-2*min(leg,second)*shared)
                slack = min(stocks[kind]-leg,stocks[other]-leg-between)
                if slack < 0 or 2*between > 100:
                    continue
                rank = (slack,-leg-between,-leg)
                if best is None or rank>best:
                    best,choices = rank,set()
                if rank==best:
                    choices.update((kind,m) for m in first)
        for entry in choices:
            votes[entry] = votes.get(entry,0.) + 1/((1+error)**2*len(choices))
    return votes


class Relay(RangeRoads):
    def _fallback_move(self,view,goal,stops):
        old = tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        demand = service_order(view,self._fallback.mode,self.RESERVE)
        default = super()._fallback_move(view,goal,stops)
        f,r = view.fuel_distance,view.repair_distance
        if demand is None or not f or not r or "несовместимого" in self.activity:
            return default
        if view.health >= 2*f+r and view.fuel >= 2*r+f:
            return default
        if all(p is not None and d>p for p,d in zip(old,(f,r))):
            return default
        votes = relay_votes(view,old)
        if not votes:
            return default
        kind,move = max(votes,key=lambda key:(votes[key],key[1]==default,-key[0],-tuple(Move).index(key[1])))
        mode = (Mode.FUEL,Mode.REPAIR)[kind]
        if move != default:
            self._fallback.mode = mode
            self._fallback.returning = False
            self.activity = "Первый заезд к паре — "+mode.value
        return move
