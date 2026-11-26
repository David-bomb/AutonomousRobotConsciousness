"""Выбрать ход, сохраняющий достижимость станции при любом совместимом выходе."""
from collections import defaultdict
from functools import lru_cache
from math import inf

from .frontier_automaton import _paths
from .grid import cell, destination
from .interface import Move
from .proven_approach import compatible_exits, service_field
from .sight_automaton import window_unknown
from .street_trunk import StreetTrunkReserve


@lru_cache(maxsize=128)
def arrival_bounds(memory, position, measured, old, prior, sites):
    known = service_field(memory, sites)
    approaches = set()
    if known.get(position) != measured:
        approaches = {(p, prefix) for p, _, prefix, _ in compatible_exits(memory, position, measured, old, prior, sites)}
    bounds = {}
    for move in Move:
        q = destination(memory, position, move)
        if q is None:
            continue
        paths = _paths(memory, q)
        # Если кратчайшая дорога впервые выходит из памяти через p, от p до
        # сервиса остаётся ровно measured-prefix. До p можно вернуться по памяти.
        upper = max((paths[p][0] + measured - prefix for p, prefix in approaches), default=inf)
        bounds[move] = 1 + min(known.get(q, inf), upper)
    return bounds


class CoveredApproach(StreetTrunkReserve):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        original = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        stocks = view.fuel, view.health
        budget = min(stocks[kind], stocks[1-kind] - 1)
        distance = readings[kind]
        if distance is None or not 0 <= budget - distance < 2:
            return original
        bounds = arrival_bounds(view.memory, view.position, distance, old[kind], prior, sites[kind])
        if bounds[original] <= budget:
            return original
        feasible = [m for m, length in bounds.items() if length <= budget]
        if not feasible:
            return original
        return min(feasible, key=lambda m: (bounds[m], m not in options,
                   -len(window_unknown(view.memory, destination(view.memory, view.position, m)))))


def visible_gates(memory, exits, window):
    """Сколько разных направлений можно проверить наблюдением необходимых клеток прохода."""
    groups = defaultdict(set)
    for p, q, _, first in exits:
        cells = {q}
        if q[0] != p[0] and q[1] != p[1]:
            cells.update(((p[0], q[1]), (q[0], p[1])))
        needed = frozenset(point for point in cells if cell(memory, *point) == "?")
        if needed:
            groups[frozenset(first)].add(needed)
    if len(groups) < 2:
        return 0.
    return sum(sum(gate <= window for gate in gates) / len(gates) for gates in groups.values())


class ObservedExit(CoveredApproach):
    """Пока хватает на одну пробу, открыть клетки, различающие возможные подходы."""
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        original = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        stocks = view.fuel, view.health
        budget = min(stocks[kind], stocks[1-kind] - 1)
        measured = readings[kind]
        if measured is None or budget - measured not in (1, 2):
            return original
        args = view.memory, view.position, measured, old[kind], prior, sites[kind]
        bounds = arrival_bounds(*args)
        exits = compatible_exits(*args)
        def information(move):
            q = destination(view.memory, view.position, move)
            return visible_gates(view.memory, exits, window_unknown(view.memory, q))
        feasible = [m for m, length in bounds.items() if length <= budget]
        choice = max(feasible, key=lambda m: (information(m), -bounds[m], m == original), default=original)
        return choice if information(choice) > information(original) else original
