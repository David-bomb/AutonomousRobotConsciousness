"""Запасные окна, до которых можно доехать по памяти с пополнением в пути."""
from collections import defaultdict
from functools import lru_cache

from .frontier_automaton import _arrive, _service_states
from .grid import cell, locations
from .second_window import ReturnWindows, closed_probability


@lru_cache(maxsize=128)
def backup_stocks(memory, origin, fuel, health, points, *, include_origin=False):
    stations = tuple((p,k) for k in "FR" for p in locations(memory,k))
    arrivals = defaultdict(set)
    for _,f,h,_,roads in _service_states(memory,origin,fuel,health,stations):
        for p in points:
            if (p == origin and not include_origin) or p not in roads:
                continue
            length = roads[p][0]
            stocks = _arrive(cell(memory,*p),length,f,h) if length else (f,h)
            if stocks:
                arrivals[p].add(stocks)
    return {p:tuple(sorted((f,h) for f,h in stocks if not any(
                a>=f and b>=h and (a,b)!=(f,h) for a,b in stocks))) for p,stocks in arrivals.items()}


class ServicedWindows(ReturnWindows):
    additional_windows = 1

    def backup(self, view, goal, a, curves):
        def best_after(point, fuel, health, remaining, tried):
            options = backup_stocks(view.memory,point,fuel,health,tuple(curves))
            scores = [0.]
            for p,stocks in options.items():
                if p in tried:
                    continue
                for f,h in stocks:
                    score = curves[p][min(f,h)]
                    if remaining > 1:
                        score += best_after(p,f,h,remaining-1,tried|{p})
                    scores.append(score)
            return closed_probability(view.memory,point,goal)*max(scores)
        return best_after(a.point,a.fuel,a.health,self.additional_windows,{a.point})


class ThreeWindows(ServicedWindows):
    additional_windows = 2
