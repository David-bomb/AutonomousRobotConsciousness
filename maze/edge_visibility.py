"""Оценка доступности хода по всем клеткам, необходимым для его выполнения."""
from functools import lru_cache, partial

import numpy as np
from scipy.sparse import coo_matrix

from .cell_hypotheses import occupancy
from .road_hypotheses import lattice
from .simple_exit_paths import SimpleExitPaths, simple_exit_probabilities


def joint_chances(free, edges):
    u, v, a, b = edges
    weights = free[u] * free[v]
    diagonal = (a != u) & (b != u)
    weights[diagonal] *= free[a[diagonal]] * free[b[diagonal]]
    return weights


@lru_cache(maxsize=128)
def joint_graph(memory, goal):
    board, chance = occupancy(memory)
    free = np.where(board == b"?", chance, board != b"#")
    free[goal[1] * len(memory[0]) + goal[0]] = 1.
    edges = lattice(len(memory[0]), len(memory))
    weights = joint_chances(free, edges)
    usable = weights > 0
    graph = coo_matrix((weights[usable], tuple(edges[:2, usable])), shape=(len(board), len(board))).tocsr()
    return graph + graph.T


class JointEdgePaths(SimpleExitPaths):
    probabilities = staticmethod(partial(simple_exit_probabilities, graph_builder=joint_graph))
