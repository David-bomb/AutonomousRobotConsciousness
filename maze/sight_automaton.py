"""Эксперимент E-0009. Выбор окна по различимости подходов к сервису."""
from dataclasses import dataclass, replace

from .frontier_automaton import CONFIRMED_GOAL
from .frontier_reserve import ReserveFrontier
from .grid import cell, destination
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import known_paths, opposite, possible_directions, service_order


@dataclass(frozen=True)
class Probe:
    move: Move
    detour: bool
    distinguishing_cells: tuple


def window_unknown(memory, point):
    x, y = point
    return {(i, j) for i in range(x - 1, x + 2) for j in range(y - 1, y + 2)
            if cell(memory, i, j) == "?"}


def with_wall(memory, point):
    x, y = point
    row = memory[y]
    return memory[:y] + (row[:x] + "#" + row[x + 1:],) + memory[y + 1:]


class SightRoute(ReserveFrontier):
    allow_detour = False

    def __init__(self):
        super().__init__()
        self._return_from_probe = False

    def choose(self, view):
        if view.last_move is None:
            self._return_from_probe = False
        returning = self._return_from_probe
        proposed = super().choose(view)
        if returning:
            self._return_from_probe = False
            if view.moved and self.activity != CONFIRMED_GOAL:
                self._fallback.returning = False
                self.activity = "Возврат после обзора"
                return opposite(view.last_move)
        return proposed

    def _fallback_move(self, view, goal, stops):
        old_mode = self._fallback.mode
        previous = self._fallback.last_lengths
        returning = self._fallback.returning
        base = super()._fallback_move(view, goal, stops)
        if returning or self._return_from_probe:
            return base
        if self.activity not in (Mode.FUEL.value, Mode.REPAIR.value):
            return base
        service = service_order(view, old_mode, self.RESERVE)
        if service is None or self.activity != service.mode.value:
            return base
        old = previous.get(service.mode) if view.moved else None
        if old is not None and old < service.distance:
            return base
        if view.fuel_distance is None or view.repair_distance is None:
            return base
        if min(view.fuel - view.fuel_distance, view.health - view.repair_distance) < 2:
            return base
        other = view.health if service.mode == Mode.FUEL else view.fuel
        if other <= service.distance + 2:
            return base
        paths = known_paths(view.memory, view.position)
        if any(paths.get(site, (None,))[0] == service.distance for site in service.sites):
            return base
        probe = self._select_probe(view, service, old, base)
        if probe is None:
            return base
        self._return_from_probe = probe.detour
        self._fallback.returning = False
        self.activity = "Боковой обзор" if probe.detour else "Обзор на пути к сервису"
        return probe.move

    def _select_probe(self, view, service, old, base):
        options = possible_directions(view, service, old).keys()
        if base not in options or len(options) < 2:
            return None
        windows = {}
        for move in Move:
            if move not in options and not self.allow_detour:
                continue
            point = destination(view.memory, view.position, move)
            if point is not None and cell(view.memory, *point) not in "FRB":
                windows[move] = window_unknown(view.memory, point)
        if base not in windows or len(windows) < 2:
            return None

        effects = {}
        for point in set.union(*windows.values()):
            hypothetical = replace(view, memory=with_wall(view.memory, point))
            remaining = possible_directions(hypothetical, service, old).keys()
            if remaining and len(remaining) < len(options):
                effects[point] = options - remaining

        def score(move):
            cells = windows[move] & effects.keys()
            removed = set().union(*(effects[p] for p in cells))
            return len(removed), len(cells), move in options, move == base

        best = max(windows, key=score)
        if best == base:
            return None
        return Probe(best, best not in options, tuple(sorted(windows[best] & effects.keys())))


class SightDetour(SightRoute):
    allow_detour = True
