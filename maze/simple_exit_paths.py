"""Оценить длину пути от цели к первому окну после удаления петель в предполагаемой дороге."""
from bisect import bisect_right
from functools import lru_cache
from itertools import accumulate
from random import Random

import numpy as np

from .exit_flow import ExitCurrent, eventual_exits
from .mission_order import MissionOrderReserve
from .passage_flow import passage_graph


def walk_lengths(rows, links, cumulative, source, sinks, draws, *, trace=False):
    rng = Random(42)
    last_exit = [-1] * (len(rows) - 1)
    for _ in range(draws):
        p = source
        while p not in sinks:
            cdf = cumulative[p]
            q = links[rows[p] + min(bisect_right(cdf, rng.random() * cdf[-1]), len(cdf) - 1)]
            last_exit[p] = q
            p = q
        terminal = p
        # Последний выход из клетки даёт маршрут после удаления всех петель.
        p, length = source, 0
        path = [source] if trace else None
        while p not in sinks and (trace or length <= 100):
            p = last_exit[p]
            length += 1
            if trace:
                path.append(p)
        yield terminal, tuple(path) if trace else length


def sampled_walks(memory, goal, points, graph_builder, *, trace=False):
    graph = graph_builder(memory, goal)
    width = len(memory[0])
    source = goal[0] + width * goal[1]
    sinks = {x + width * y: (x, y) for x, y in points}
    links = graph.indices.tolist()
    weights = graph.data.tolist()
    rows = graph.indptr.tolist()
    cumulative = [tuple(accumulate(weights[a:b])) for a, b in zip(rows, rows[1:])]
    for p, result in walk_lengths(rows, links, cumulative, source, sinks, 64, trace=trace):
        yield sinks[p], result


@lru_cache(maxsize=32)
def simple_exit_probabilities(memory, goal, points, graph_builder=passage_graph):
    # Точная доля первого попадания в окно и оптимистичная отсечка по длине.
    initial = eventual_exits(memory, goal, points, graph_builder)
    if goal in points or not any(row[-1] for row in initial.values()):
        return initial
    sampled = dict.fromkeys(points, 0)
    lengths = {p: np.zeros(101) for p in points}
    for terminal, length in sampled_walks(memory, goal, points, graph_builder):
        sampled[terminal] += 1
        if length <= 100:
            lengths[terminal][length] += 1
    return {p: row * lengths[p].cumsum() / sampled[p] if sampled[p] else row for p, row in initial.items()}


class SimpleExitPaths(ExitCurrent):
    probabilities = staticmethod(simple_exit_probabilities)


class MissionSimplePaths(MissionOrderReserve, SimpleExitPaths):
    """Выбирает первую станцию по продолжению поездки, а выезд к цели — по длинам путей без петель."""
