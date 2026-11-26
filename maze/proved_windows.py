"""Выводить свойства неизвестных клеток из замеров перед последним выездом."""
from dataclasses import replace
from functools import lru_cache

from .deduced_roads import deductions
from .frontier_reserve import _lengths, _optimistic
from .grid import locations
from .pilot_automaton import Mode
from .resource_automaton import confirmed_route, known_paths
from .second_window import SecondWindow


@lru_cache(maxsize=96)
def proved_grid(memory, goal, position, readings, prior, old):
    x, y = goal
    if memory[y][x] == "?":
        memory = memory[:y] + (memory[y][:x] + "B" + memory[y][x+1:],) + memory[y+1:]
    sites = tuple(tuple(locations(memory, k)) for k in "FR")
    optimistic = _optimistic(memory)

    def informative(origin, sites, value):
        if value is None or origin is None or not sites:
            return None
        # Если длина одинакова при открытии и закрытии всех неизвестных клеток,
        # это наблюдение уже выполняется при любой достройке.
        for grid in (memory, optimistic):
            distances = _lengths(grid, origin)
            if min(distances.get(p, float("inf")) for p in sites) != value:
                return value
        return None

    current = tuple(informative(position, s, d) for s, d in zip(sites, readings))
    previous = tuple(informative(prior, s, d) for s, d in zip(sites, old))
    if not any(current + previous):
        return memory
    return deductions(memory, position, goal, current, prior, previous, sites)


class ProvedFinish(SecondWindow):
    plan_with_proofs = False

    def _reconsider_sortie(self, view, goal, stops, planned):
        previous = None
        if view.moved:
            dx, dy = view.last_move.value
            previous = view.position[0]-dx, view.position[1]-dy
        old = tuple(self._fallback.last_lengths.get(mode) for mode in (Mode.FUEL, Mode.REPAIR))
        memory = proved_grid(view.memory, goal, view.position, (view.fuel_distance,view.repair_distance), previous, old)
        if memory != view.memory:
            proposed = replace(view, memory=memory)
            route = confirmed_route(proposed, goal, known_paths(memory, view.position))
            if route:
                self._fallback.mode, move = route
                self._fallback.returning = False
                self._fallback.last_lengths = {Mode.FUEL:view.fuel_distance, Mode.REPAIR:view.repair_distance}
                self.activity = "Путь к финишу подтверждён длинами до станций"
                return move
            if self.plan_with_proofs:
                move = super()._reconsider_sortie(proposed, goal, stops, planned)
                if move is not None:
                    return move
        return super()._reconsider_sortie(view, goal, stops, planned)


class ProvedWindows(ProvedFinish):
    plan_with_proofs = True
