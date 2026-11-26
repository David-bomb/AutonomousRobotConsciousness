"""Сопоставление неизвестного продолжения дороги с длиной до сервиса."""
from collections import deque
from functools import lru_cache
from math import exp

from .frontier_automaton import _paths
from .frontier_reserve import ReserveFrontier
from .grid import cell, locations
from .interface import Move
from .pilot_automaton import Mode, separation
from .resource_automaton import Service, possible_directions, service_order


@lru_cache(maxsize=128)
def exits(grid, origin):
    """Любая кратчайшая дорога имеет кратчайший уже известный префикс."""
    distances = {origin: 0}
    directions = {origin: set()}
    queue = deque([origin])
    ends = []
    while queue:
        p = queue.popleft()
        length = distances[p]+1
        for move in Move:
            dx, dy = move.value
            q = p[0]+dx, p[1]+dy
            first = directions[p] if p != origin else {move}
            needed = [cell(grid, *q)]
            if dx and dy:
                needed.extend((cell(grid, q[0], p[1]), cell(grid, p[0], q[1])))
            if "#" in needed:
                continue
            if "?" in needed:
                if p != origin:
                    ends.append((q, move.value, length, tuple(first), needed.count("?")))
                continue
            if q not in distances:
                distances[q], directions[q] = length, set(first)
                queue.append(q)
            elif distances[q] == length:
                directions[q].update(first)
    return tuple(ends)


def evidence(view, service, old):
    options = possible_directions(view, service, old)
    if not options:
        return {}
    paths = _paths(view.memory, view.position)
    exact = {paths[p][1] for p in service.sites
             if p in paths and paths[p][0] == service.distance}
    if exact:
        return {m: float(m in exact) for m in options}
    votes = dict.fromkeys(options, 0.0)
    for q, (dx, dy), length, first, hidden in exits(view.memory, view.position):
        remaining = service.distance - length
        if remaining < 0:
            continue
        score = 0.
        for site in service.sites:
            if separation(q, site) > remaining:
                continue
            # Гипотеза о продолжении: до первого поворота дорога может
            # сохранить направление. При большем остатке допустим больший обход.
            delta_x = site[0] - q[0] - dx * min(remaining, 1)
            delta_y = site[1] - q[1] - dy * min(remaining, 1)
            variance = 1 + remaining
            weight = exp(-(delta_x**2 + delta_y**2)/(2*variance)) / (variance * 2**(hidden-1))
            score = max(score, weight)
        for move in first:
            if move in votes:
                votes[move] = max(votes[move], score)
    peak = max(votes.values())
    return {m: value/peak for m, value in votes.items()} if peak else {}


class Passage(ReserveFrontier):
    joint = False
    service_evidence = staticmethod(evidence)

    def _fallback_move(self, view, goal, stops):
        previous = self._fallback.last_lengths.copy()
        active = self._fallback.mode
        was_returning = self._fallback.returning
        service = service_order(view, active, self.RESERVE)
        default = super()._fallback_move(view, goal, stops)
        if service is None or was_returning or "несовместимого" in self.activity:
            return default
        old = previous.get(service.mode) if view.moved else None
        if old is not None and old < service.distance:
            return default
        options = possible_directions(view, service, old)
        if len(options) < 2 or default not in options:
            return default
        roads = _paths(view.memory, view.position)
        if any(roads.get(p, (None,))[0] == service.distance for p in service.sites):
            return default
        scores = self.service_evidence(view, service, old)
        if not scores:
            return default
        other_scores = {}
        if self.joint:
            fuel_first = service.mode == Mode.FUEL
            mode = Mode.REPAIR if fuel_first else Mode.FUEL
            other_d = view.repair_distance if fuel_first else view.fuel_distance
            stock = view.health if fuel_first else view.fuel
            if other_d is not None and 0 < other_d <= stock:
                sites = tuple(locations(view.memory, "R" if fuel_first else "F"))
                other = Service(mode, stock, other_d, sites)
                other_scores = self.service_evidence(view, other, previous.get(mode) if view.moved else None)
        def rank(move):
            # У другого сервиса вспомогательный голос. Отсутствие общей
            # дороги совместимо с заездом по очереди через текущую клетку.
            return scores[move] * (1+other_scores.get(move, 0)), move == default
        move = max(scores, key=rank)
        if move != default:
            self.activity = "Выход с известной дороги — " + service.mode.value
        return move


class JointPassage(Passage):
    joint = True
