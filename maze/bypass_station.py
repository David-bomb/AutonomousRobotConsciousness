"""Проверять, сохраняет ли ближайшая станция возможность продолжить поездку."""
from functools import lru_cache
from math import inf, isfinite

from .frontier_automaton import _arrive
from .frontier_reserve import _lengths, _optimistic
from .grid import destination
from .interface import Move
from .last_mission import trip_length
from .mission_order import first_stop_models
from .pilot_automaton import Mode
from .specific_paths import SpecificPaths


@lru_cache(maxsize=96)
def bypass_evidence(view, goal, old, kind):
    width = len(view.memory[0])
    index = lambda p: p[1]*width + p[0]
    origin = index(view.position)
    measured = (view.fuel_distance, view.repair_distance)[kind]
    roads = _lengths(view.memory, view.position)
    lower = _lengths(_optimistic(view.memory), view.position)
    moves = {m: index(q) for m in Move if (q := destination(view.memory, view.position, m)) is not None}
    scores = dict.fromkeys(moves, 0.)
    nearer = farther = 0.
    for weight, fields, sites, _ in first_stop_models(view, goal, old):
        candidates = [j for j, (k, _) in enumerate(sites) if k == "FR"[kind]]
        nearest = min(fields[j, origin] for j in candidates)
        close_works = False
        beyond = set()
        for j in candidates:
            k, target = sites[j]
            if not isfinite(fields[j, origin]):
                continue
            point = target % width, target // width
            length = max(measured, lower.get(point, inf), min(roads.get(point, inf),
                         measured + fields[j, origin] - nearest))
            if not isfinite(length) or length <= 0:
                continue
            after = _arrive(k, int(length), view.fuel, view.health)
            if after is None or not isfinite(trip_length(target, after, sites, fields, {})):
                continue
            if length == measured:
                close_works = True
                break
            beyond.update(m for m, q in moves.items() if fields[j, q] == fields[j, origin]-1)
        if close_works:
            nearer += weight
        elif beyond:
            farther += weight
            for m in beyond:
                scores[m] += weight / len(beyond)
    return nearer, farther, scores


class BypassStation(SpecificPaths):
    unanimous = False

    def _fallback_move(self, view, goal, stops):
        modes = (Mode.FUEL, Mode.REPAIR)
        old = tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in modes)
        parent = super()._fallback_move(view, goal, stops)
        if self._fallback.mode not in modes:
            return parent
        kind = modes.index(self._fallback.mode)
        own, other = (view.fuel, view.health) if kind == 0 else (view.health, view.fuel)
        reading = (view.fuel_distance, view.repair_distance)[kind]
        limit = min(own, other-1)
        if reading is None or not 0 < reading < limit:
            return parent
        floors = _lengths(_optimistic(view.memory), view.position)
        if sum(0 < floors.get(p, inf) <= limit for p, k in stops if k == "FR"[kind]) < 2:
            return parent
        near, far, scores = bypass_evidence(view, goal, old, kind)
        if far <= near or (self.unanimous and near):
            return parent
        chosen = max(scores, key=lambda m: (scores[m], m == parent))
        if chosen != parent:
            self.activity = "Дальняя станция с продолжением до финиша"
            self._fallback.returning = False
        return chosen


class UnanimousBypass(BypassStation):
    unanimous = True
