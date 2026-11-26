"""Два окна могут зависеть от одного и того же скрытого прохода."""
from functools import lru_cache
from math import prod

from .cell_hypotheses import occupancy
from .frontier_reserve import _lengths, _optimistic
from .grid import cell, destination
from .interface import Move
from .second_window import SecondWindow, closed_probability


@lru_cache(maxsize=512)
def opening_cells(memory, point):
    x, y = point
    cells = set()
    for m in Move:
        q = destination(_optimistic(memory), point, m)
        if q is None or cell(memory, *q) != "?":
            continue
        dx, dy = m.value
        if dx and dy and (cell(memory, x+dx, y) == "?" or cell(memory, x, y+dy) == "?"):
            # Открытие неизвестного угла само по себе даёт прямой ход к нему.
            continue
        cells.add(q)
    return frozenset(cells)


@lru_cache(maxsize=2048)
def closure_given_exit(memory, goal, first, second):
    a, b = opening_cells(memory, first), opening_cells(memory, second)
    closed = closed_probability(memory, first, goal)
    if a.isdisjoint(b):
        return closed
    _, occupancy_prob = occupancy(memory)
    width = len(memory[0])
    def walls(cells):
        return prod(0. if p == goal else 1. - occupancy_prob[p[1]*width+p[0]] for p in sorted(cells))
    # P(первое закрыто | второе открыто); одна и та же клетка учитывается один раз.
    return (walls(a) - walls(a | b)) / (1. - walls(b))


class OverlappingWindows(SecondWindow):
    def backup(self, view, goal, a, curves):
        lengths = _lengths(view.memory, a.point)
        stock = min(a.fuel, a.health)
        return max((closure_given_exit(view.memory, goal, a.point, p) * curve[stock - lengths[p]]
                    for p, curve in curves.items() if p != a.point and lengths.get(p, 101) < stock), default=0.)
