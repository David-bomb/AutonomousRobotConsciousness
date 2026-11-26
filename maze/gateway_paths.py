"""Посчитать скрытый путь до въезда на известную дорогу, затем открытый путь до окна обзора."""
from functools import lru_cache

from .first_window import MarginWindow, first_events
from .frontier_automaton import _frontiers
from .frontier_reserve import _lengths
from .service_contact import road_gates
from .simple_exit_paths import SimpleExitPaths


@lru_cache(maxsize=64)
def approaches_from_gates(memory, gates, windows):
    approaches = {point: [] for point in windows}
    for gate in gates:
        if gate in windows:
            approaches[gate].append((gate, 0))
        else:
            for point, length, _ in first_events(memory, gate, windows):
                approaches[point].append((gate, length))
    return approaches


class GatewayPaths(SimpleExitPaths):
    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        if len({a.point for a in choices}) < 2 or goal in _lengths(view.memory, view.position):
            return super().pick(view, goal, choices)
        gates = road_gates(view.memory, view.position)
        if not gates:
            return super().pick(view, goal, choices)
        windows = frozenset(p for p, _, _ in first_events(
            view.memory, view.position, frozenset(_frontiers(view.memory))))
        approaches = approaches_from_gates(view.memory, gates, windows)
        curves = self.probabilities(view.memory, goal, gates)
        scores = []
        for arrival in choices:
            stock = min(arrival.fuel, arrival.health)
            scores.append(sum(curves[p][stock - leg] for p, leg in approaches[arrival.point] if leg <= stock))
        best = max(scores)
        if not best:
            return super().pick(view, goal, choices)
        return MarginWindow.pick(self, view, goal, [a for a, score in zip(choices, scores) if score == best])
