"""Оценивать объезды к цели по длинам путей между другими наблюдавшимися клетками."""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .exit_flow import eventual_exits
from .grid import cell
from .road_hypotheses import lattice
from .simple_exit_paths import simple_exit_probabilities
from .specific_paths import SpecificPaths


@lru_cache(maxsize=32)
def observed_lengths(memory):
    width, height = len(memory[0]), len(memory)
    board = np.frombuffer("".join(memory).encode(), dtype="S1")
    # Отдельно выданные координаты сервисов не являются случайными пробами
    # дорожного покрытия. В длинах путей сами станции остаются проезжими.
    targets = np.flatnonzero(board == b".")
    starts = targets
    if len(starts) > 64:
        starts = np.random.default_rng(42).choice(starts, size=64, replace=False)
    edges = lattice(width, height)
    def lengths(allowed):
        u, v = edges[:2, allowed[edges].all(axis=0)]
        graph = coo_matrix((np.ones(len(u)), (u, v)), shape=(len(board), len(board))).tocsr()
        field = dijkstra(graph, indices=starts, directed=False, unweighted=True, limit=100)
        return np.minimum(field[:, targets], 101).astype(np.uint8)
    lower = lengths(board != b"#")
    upper = lengths((board != b"#") & (board != b"?"))
    dx = abs(starts[:, None] % width - targets % width)
    dy = abs(starts[:, None] // width - targets // width)
    stride = max(width, height)
    offset = np.minimum(dx, dy) * stride + np.maximum(dx, dy)
    return offset, lower, upper


def adjusted_curve(prior, share, reachable, lower, upper):
    if not len(lower) or not share:
        return prior
    # Если даже оптимистичный путь длиннее запаса, пример даёт отказ.
    # Если известная дорога укладывается в запас, пример даёт успех.
    # В остальных случаях сохраняется вероятность исходной модели.
    possible = np.bincount(lower, minlength=102)[:101].cumsum()
    sure = np.bincount(upper, minlength=102)[:101].cumsum()
    result = (sure * share + (1 + possible - sure) * prior) / (1 + len(lower))
    return np.where(reachable, result, 0.)


@lru_cache(maxsize=32)
def measured_probabilities(memory, goal, windows):
    curves = simple_exit_probabilities(memory, goal, windows)
    if cell(memory, *goal) in "FR" or goal in windows:
        return curves
    shares = eventual_exits(memory, goal, windows)
    offset, lower, upper = observed_lengths(memory)
    stride = max(len(memory), len(memory[0]))
    result = {}
    for p, prior in curves.items():
        dx, dy = sorted((abs(p[0] - goal[0]), abs(p[1] - goal[1])))
        match = offset == dx * stride + dy
        result[p] = adjusted_curve(prior, shares[p][-1], shares[p] > 0, lower[match], upper[match])
    return result


class MeasuredDetours(SpecificPaths):
    probabilities = staticmethod(measured_probabilities)
