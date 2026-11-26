"""Варианты единой проверки допустимости для всех режимов frontier."""
from functools import lru_cache

from .frontier_automaton import CONFIRMED_GOAL, FrontierAutomaton, _arrive, _paths
from .frontier_reserve import ReserveFrontier, _optimistic, _relaxed_route, _service_floors
from .grid import cell, destination, locations
from .interface import Move, Observation


class Guard:
    CHECK_MARGIN = 6
    USE_CERTIFICATE = False

    def _critical(self, view):
        if not self.USE_CERTIFICATE:
            return ()
        return tuple((kind, d) for kind, stock, d in (
            ("F", view.fuel, view.fuel_distance), ("R", view.health, view.repair_distance))
            if d is not None and 0 < d <= stock <= d + 1)

    def choose(self, view: Observation) -> Move:
        previous = {} if view.last_move is None else self._fallback.last_lengths.copy()
        proposed = super().choose(view)
        margins = (stock - distance for stock, distance in (
            (view.fuel, view.fuel_distance), (view.health, view.repair_distance))
            if distance is not None)
        if self.activity == CONFIRMED_GOAL or min(margins, default=100) > self.CHECK_MARGIN:
            return proposed
        x, y = view.position
        goal = x + view.dx, y + view.dy
        optimistic = _optimistic(view.memory)
        services = tuple((p, kind) for kind in "FR" for p in locations(view.memory, kind))
        targets = (*services, (goal, "B"))
        critical = self._critical(view)
        possible = []
        for move in Move:
            point = destination(view.memory, view.position, move)
            if point is None:
                continue
            kind = "B" if point == goal else cell(view.memory, *point)
            stocks = _arrive(kind, 1, view.fuel, view.health)
            floors = _service_floors(view, previous, point)
            if stocks is None or not _relaxed_route(optimistic, point, goal, *stocks, targets, floors):
                continue
            if move == proposed and not critical:
                return proposed
            certified = bool(critical) and all(
                any(k == target_kind and _paths(view.memory, site).get(point, (-1,))[0] == d - 1
                    for site, k in services) for target_kind, d in critical)
            slack = min(stock - floors.get(k, 0) for k, stock in zip("FR", stocks))
            possible.append(((point == goal, certified, move == proposed, kind in "FR", slack), move))
        choice = max(possible, key=lambda row: row[0])[1] if possible else proposed
        if choice != proposed:
            self._fallback.returning = False
            self.activity = "Единая проверка продолжения"
        return choice


class GuardedBase(Guard, FrontierAutomaton):
    pass


class GuardedReserve(Guard, ReserveFrontier):
    pass


class GuardedProbe(GuardedReserve):
    def _may_rank(self, view, service):
        margins = (stock - d for stock, d in (
            (view.fuel, view.fuel_distance), (view.health, view.repair_distance)) if d is not None)
        return min(margins, default=100) >= self.RESERVE


class GuardedEvidence(GuardedProbe):
    USE_CERTIFICATE = True

@lru_cache(maxsize=512)
def _after_escape(grid, station, goal, stocks, stops, floors):
    return _relaxed_route(_optimistic(grid), station, goal, *stocks, (*stops, (goal, "B")), dict(floors))


class EscapeContinuation:
    def _accept_escape(self, view, goal, stops, station, total, stocks):
        floors = tuple((kind, max(0, d - total)) for kind, d in
                       (("F", view.fuel_distance), ("R", view.repair_distance)) if d is not None)
        return _after_escape(view.memory, station, goal, stocks, stops, floors)

class GuardedExcursion(EscapeContinuation, GuardedEvidence):
    pass


class GuardedExcursionOnly(EscapeContinuation, GuardedProbe):
    pass
