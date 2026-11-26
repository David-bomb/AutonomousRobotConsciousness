"""Оценить запас на ошибочный выезд по обследованным тупикам текущего лабиринта."""
from collections import Counter
from functools import lru_cache

import numpy as np

from .frontier_automaton import _frontiers
from .grid import cell, destination
from .interface import Move
from .second_window import SecondWindow
from .simple_exit_paths import simple_exit_probabilities


@lru_cache(maxsize=64)
def branch_lengths(memory, goal):
    """Длины проверенных тупиковых ходов от развилок до конца и обратно."""
    boundary = frozenset(_frontiers(memory))
    roads = {(x, y) for y, row in enumerate(memory) for x, c in enumerate(row) if c not in "#?"}
    links = {p: tuple(q for m in Move if (q := destination(memory, p, m)) is not None) for p in roads}
    # Одна проба выбирает один выход из полностью видимой развилки.
    # Выходы, для которых тупик не подтверждён, не уменьшают расчётный запас.
    arms = sum(len(edges) for p, edges in links.items() if p not in boundary and len(edges) > 2)
    lengths = Counter()
    for leaf, neighbors in links.items():
        if len(neighbors) != 1 or leaf in boundary or leaf == goal or cell(memory, *leaf) in "FR":
            continue
        previous, current = leaf, neighbors[0]
        length = 1
        while current not in boundary and current != goal and cell(memory, *current) not in "FR":
            exits = links[current]
            if len(exits) > 2:
                lengths[2*length] += 1
                break
            if len(exits) != 2:
                break
            previous, current = current, next(q for q in exits if q != previous)
            length += 1
    lengths[0] = arms - sum(lengths.values()) if arms else 1
    return tuple(sorted(lengths.items()))


@lru_cache(maxsize=64)
def allowed_curves(memory, goal, points):
    direct = simple_exit_probabilities(memory, goal, points)
    if cell(memory, *goal) in "FR":
        return direct
    lengths = branch_lengths(memory, goal)
    if len(lengths) == 1:
        return direct
    weights = np.zeros(101)
    total = sum(n for _, n in lengths)
    for d, n in lengths:
        if d <= 100:
            weights[d] = n / total
    return {p: np.convolve(curve, weights)[:101] for p, curve in direct.items()}


class BackupAllowance(SecondWindow):
    """После осмотра тупика оставить запас на одну пробу на продолжении запасного пути."""
    def backup(self, view, goal, a, curves):
        adjusted = allowed_curves(view.memory, goal, tuple(curves))
        return super().backup(view, goal, a, adjusted)


class BranchAllowance(SecondWindow):
    """Одинаково учесть разведку в первом и в запасном выезде."""
    probabilities = staticmethod(allowed_curves)
