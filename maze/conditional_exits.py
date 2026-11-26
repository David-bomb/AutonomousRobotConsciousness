"""Для каждого возможного окна отдельно оценить длину пути при условии первого выхода именно к нему."""
from bisect import bisect_right
from collections import defaultdict
from functools import lru_cache
from itertools import accumulate

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .first_window import first_events
from .frontier_automaton import _frontiers
from .passage_flow import passage_graph
from .simple_exit_paths import SimpleExitPaths, walk_lengths


@lru_cache(maxsize=8)
def hit_law(memory, goal, points):
    graph = passage_graph(memory, goal)
    width = len(memory[0])
    source = goal[0] + goal[1] * width
    ids = np.array([x + y * width for x, y in points])
    prob = np.zeros((graph.shape[0], len(ids)))
    prob[ids, np.arange(len(ids))] = 1.
    if source not in ids:
        inside = np.setdiff1d(np.arange(graph.shape[0]), ids)
        _, labels = connected_components(graph[inside][:, inside], directed=False)
        inside = inside[labels == labels[np.searchsorted(inside, source)]]
        exits = graph[inside][:, ids]
        if exits.nnz:
            degree = np.asarray(graph[inside].sum(axis=1)).ravel()
            laplacian = diags(degree) - graph[inside][:, inside]
            prob[inside] = np.clip(splu(laplacian.tocsc()).solve(exits.toarray()), 0., 1.)
    return graph, prob, source, ids


@lru_cache(maxsize=64)
def conditional_lengths(memory, goal, points, j):
    graph, hit, source, ids = hit_law(memory, goal, points)
    if not hit[source, j]:
        return ()
    # Вероятность перехода пропорциональна проводимости ребра, умноженной
    # на вероятность попадания в нужное окно из следующей клетки.
    conditioned = graph.copy()
    conditioned.data *= hit[graph.indices, j]
    conditioned.eliminate_zeros()
    weights = conditioned.data.tolist()
    rows = conditioned.indptr.tolist()
    cumulative = [tuple(accumulate(weights[a:b])) for a, b in zip(rows, rows[1:])]
    lengths = walk_lengths(rows, conditioned.indices.tolist(), cumulative, source, {int(ids[j])}, 32)
    return tuple(sorted(d for _, d in lengths))


class ConditionalExits(SimpleExitPaths):
    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        by_point = defaultdict(list)
        for a in choices:
            by_point[a.point].append(a)
        if len(by_point) < 2 or goal in by_point:
            return super().pick(view, goal, choices)
        points = tuple(sorted(p for p, _, _ in first_events(view.memory, view.position,
                                                             frozenset(_frontiers(view.memory)))))
        _, hit, source, _ = hit_law(view.memory, goal, points)
        best, winners = 0., []
        for j in np.argsort(-hit[source]):
            share = hit[source, j]
            if share <= 0 or share < best:
                break
            if points[j] not in by_point:
                continue
            lengths = conditional_lengths(view.memory, goal, points, int(j))
            for a in by_point[points[j]]:
                score = share * bisect_right(lengths, min(a.fuel, a.health)) / len(lengths)
                if score > best:
                    best, winners = score, []
                if score == best:
                    winners.append(a)
        return super().pick(view, goal, winners if best else choices)
