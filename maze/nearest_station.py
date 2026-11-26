"""Разделить неопределённость о ближайшей станции и неопределённость о дороге к ней."""
from functools import lru_cache
from math import isfinite

from .frontier_automaton import _arrive
from .frontier_reserve import _lengths, _optimistic
from .last_mission import trip_length
from .mission_order import MissionOrderReserve, first_stop_models
from .simple_exit_paths import SimpleExitPaths


@lru_cache(maxsize=128)
def station_weights(view, goal, old, kind, points):
    readings = view.fuel_distance, view.repair_distance
    after = _arrive("FR"[kind], readings[kind], view.fuel, view.health)
    if after is None:
        return {}
    width = len(view.memory[0])
    ids = {y * width + x: (x, y) for x, y in points}
    support = dict.fromkeys(points, 0.)
    for weight, fields, markers, nearest in first_stop_models(view, goal, old):
        usable = [q for q in nearest[kind] if q in ids and isfinite(trip_length(q, after, markers, fields, {}))]
        for q in usable:
            support[ids[q]] += weight / len(usable)
    return support


class NearestStation(MissionOrderReserve):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        measured = readings[kind]
        if 0 <= (view.fuel, view.health)[kind] - measured < 2 and len(sites[kind]) > 1:
            lower = _lengths(_optimistic(view.memory), view.position)
            points = tuple(p for p in sites[kind] if lower.get(p, float("inf")) <= measured)
            if len(points) > 1:
                goal = view.position[0] + view.dx, view.position[1] + view.dy
                support = station_weights(view, goal, old, kind, points)
                scores = dict.fromkeys(options, 0.)
                for point, weight in support.items():
                    if not weight:
                        continue
                    targets = list(sites)
                    targets[kind] = (point,)
                    row = self.direction_votes(view, readings, old, prior, tuple(targets))[kind]
                    total = sum(row.get(m, 0.) for m in options)
                    if total:
                        for move in options:
                            scores[move] += weight * row.get(move, 0.) / total
                if any(scores.values()):
                    return max(options, key=lambda m: (scores[m], m == default))
        return super().rank_directions(view, readings, old, prior, sites, options, kind, default)


class StationSimplePaths(NearestStation, SimpleExitPaths):
    """Различает станции одного типа при подъезде и выбирает выезд по длинам путей без петель."""
