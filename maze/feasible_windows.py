"""Оценить долю предполагаемых сетей, в которых из выбранного окна хватает запасов до цели."""
from functools import lru_cache
from math import isfinite

from .conditional_mission import possible_worlds
from .exact_worlds import pack
from .first_window import first_events
from .frontier_automaton import _frontiers
from .route_witness import geometry
from .simple_exit_paths import SimpleExitPaths


def first_hits(opened, source, terminal, shifts, limit):
    """Волна обрывается в окнах. Условия диагоналей берутся из исходной предполагаемой сети."""
    n, e, s, w = shifts
    corners = [(a, b, a(opened) & b(opened)) for a in (n, s) for b in (e, w)]
    frontier = seen = 1 << source
    found = {}
    for d in range(limit + 1):
        hits = frontier & terminal
        while hits:
            bit = hits & -hits
            found[bit.bit_length() - 1] = d
            hits ^= bit
        frontier &= ~terminal
        if not frontier:
            break
        step = n(frontier) | e(frontier) | s(frontier) | w(frontier)
        for a, b, mask in corners:
            step |= a(b(frontier)) & mask
        frontier = step & opened & ~seen
        seen |= frontier
    return found


@lru_cache(maxsize=16)
def sampled_hits(memory, goal, windows):
    width, height = len(memory[0]), len(memory)
    source = goal[1] * width + goal[0]
    terminal = sum(1 << (y * width + x) for x, y in windows)
    shifts, _ = geometry(width, height)
    markers, worlds = possible_worlds(memory, goal)
    groups = [[j for j, (kind, _) in enumerate(markers) if kind == k] for k in "FR"]
    return [(first_hits(pack(opened), source, terminal, shifts, 100),
             [fields[js].min(axis=0) if js else None for js in groups]) for opened, fields, _ in worlds]


class FeasibleWindows(SimpleExitPaths):
    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        if len({a.point for a in choices}) < 2 or any(a.point == goal for a in choices):
            return super().pick(view, goal, choices)
        windows = frozenset(p for p, _, _ in first_events(view.memory, view.position,
                                                          frozenset(_frontiers(view.memory))))
        width = len(view.memory[0])
        origin = view.position[1] * width + view.position[0]
        readings = view.fuel_distance, view.repair_distance
        stocks = [(a.point[1] * width + a.point[0], min(a.fuel, a.health)) for a in choices]
        scores = [0.] * len(choices)
        for hits, gauges in sampled_hits(view.memory, goal, windows):
            error = sum(abs(field[origin] - d) for field, d in zip(gauges, readings)
                        if field is not None and d is not None)
            if isfinite(error):
                weight = 1 / (1 + error) ** 2
                for i, (point, stock) in enumerate(stocks):
                    if hits.get(point, 101) <= stock:
                        scores[i] += weight
        best = max(scores)
        return super().pick(view, goal, [a for a, score in zip(choices, scores) if score == best])
