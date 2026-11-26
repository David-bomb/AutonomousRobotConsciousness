"""Выбор первого нового окна для движения к цели или к удалённому сервису."""
from collections import Counter
from functools import lru_cache
from math import isfinite

from .conditional_mission import required_after_stop
from .deadline_goal import goal_deadline
from .first_window import MarginWindow, first_events
from .frontier_automaton import _frontiers, _paths
from .frontier_reserve import _lengths, _optimistic
from .grid import locations
from .known_landing import continuations
from .nearest_station import NearestStation
from .remote_station import can_rejoin
from .simple_exit_paths import SimpleExitPaths


@lru_cache(maxsize=64)
def continuation_needs(memory, goal, position, readings, excluded=()):
    sites, models = continuations(memory, goal)
    width = len(memory[0])
    origin = position[1] * width + position[0]
    groups = [[j for j, (kind, _) in enumerate(sites) if kind == k] for k in "FR"]
    retained = [j for j, (_, i) in enumerate(sites) if (i % width, i // width) not in excluded]
    onward = [sites[j] for j in retained]
    by_site = {i: Counter() for _, i in sites[:-1]}
    weight_sum = 0.
    for fields, needs in models:
        error = sum(abs(min(fields[j, origin] for j in group) - d)
                    for group, d in zip(groups, readings) if group and d is not None)
        if not isfinite(error):
            continue
        weight = 1 / (1 + error) ** 2
        weight_sum += weight
        if excluded:
            needs = required_after_stop(onward, fields[retained])
        for (_, i), need in zip(onward, needs):
            if isfinite(need):
                by_site[i][max(1, int(need))] += weight
    return {(i % width, i // width): [(need, w / weight_sum) for need, w in votes.items()]
            for i, votes in by_site.items()}


def interval_mass(curve, lower, upper):
    if upper < lower:
        return 0.
    return max(0., curve[upper] - (curve[lower - 1] if lower else 0.))


class ServiceExitPaths(SimpleExitPaths):
    """За рубежом подтверждённого возврата оценивает также путь через новую станцию."""
    independent_outpost = False

    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        offered = {a.point for a in choices}
        roads = _paths(view.memory, view.position)
        stops = tuple((p, k) for k in "FR" for p in locations(view.memory, k))
        remote = [(p, k) for p, k in stops if p not in roads]
        if (len(offered) < 2 or goal in offered or not remote
                or (not self.independent_outpost and can_rejoin(view, stops, roads))):
            return super().pick(view, goal, choices)
        points = tuple(sorted(p for p, _, _ in first_events(view.memory, view.position,
                                                             frozenset(_frontiers(view.memory)))))
        direct = self.probabilities(view.memory, goal, points)
        scores = [direct[a.point][min(a.fuel, a.health)] for a in choices]
        readings = view.fuel_distance, view.repair_distance
        excluded = tuple(p for p, _ in stops if p in roads) if self.independent_outpost else ()
        needs = continuation_needs(view.memory, goal, view.position, readings, excluded)
        for site, kind in remote:
            k = "FR".index(kind)
            if readings[k] is None or not needs[site]:
                continue
            lower = _lengths(_optimistic(view.memory), site)
            if not any(lower.get(a.point, 101) <= min(a.fuel, a.health) for a in choices):
                continue
            curves = self.probabilities(view.memory, site, points)
            for i, a in enumerate(choices):
                own, other = (a.fuel, a.health) if k == 0 else (a.health, a.fuel)
                floor = max(0, readings[k] - a.length)
                score = sum(w * interval_mass(curves[a.point], floor, min(own, other - need))
                            for need, w in needs[site])
                scores[i] = max(scores[i], score)
        best = max(scores)
        return MarginWindow.pick(self, view, goal, [a for a, score in zip(choices, scores) if score == best])


class UsefulServicePaths(ServiceExitPaths):
    def pick(self, view, goal, arrivals):
        # Если ни один заезд не продлевает движение, ресурсным ориентиром
        # больше нельзя обосновать смену направления последнего выезда.
        if goal_deadline(view):
            return SimpleExitPaths.pick(self, view, goal, arrivals)
        return super().pick(view, goal, arrivals)


class OutpostExits(UsefulServicePaths):
    """После безопасной разведки рассматривает новую станцию, от которой можно доехать без прежней базы."""
    independent_outpost = True


class StationUsefulPaths(NearestStation, UsefulServicePaths):
    """При подъезде различает станции одного типа, при выезде проверяет, может ли заезд продлить поездку."""
