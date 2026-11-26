"""Доля потока к сервису по дорогам, допускающим его измеренную длину."""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve

from .cell_hypotheses import occupancy
from .grid import destination
from .interface import Move
from .road_hypotheses import lattice
from .street_trunk import StreetTrunk


@lru_cache(maxsize=128)
def passage_graph(memory, goal):
    board, chance = occupancy(memory)
    width, height = len(memory[0]), len(memory)
    free = np.where(board == b"?", chance, board != b"#")
    free[goal[1]*width+goal[0]] = 1.
    edges = lattice(width, height)
    weights = free[edges].min(axis=0)
    u, v = edges[:2, weights > 0]
    graph = coo_matrix((weights[weights > 0], (u, v)), shape=(len(board), len(board))).tocsr()
    return graph + graph.T


def flows(memory, origin, goal, sites, distance):
    if not sites or not distance:
        return {}
    width = len(memory[0])
    def index(p):
        return p[1]*width+p[0]
    graph = passage_graph(memory, goal)
    source = index(origin)
    sinks = np.array([index(p) for p in sites])
    start = dijkstra(graph, directed=False, unweighted=True, indices=source)
    finish = dijkstra(graph, directed=False, unweighted=True, indices=sinks, min_only=True)
    # На пути ровно измеренной длины каждая клетка проходит эту проверку.
    allowed = np.flatnonzero(start + finish <= distance)
    if source not in allowed:
        return {}
    reduced = graph[allowed][:, allowed]
    _, labels = connected_components(reduced, directed=False)
    nodes = allowed[labels == labels[np.searchsorted(allowed, source)]]
    targets = np.isin(nodes, sinks)
    if not targets.any():
        return {}
    interior = (nodes != source) & ~targets
    network = graph[nodes][:, nodes]
    laplacian = diags(np.asarray(network.sum(axis=1)).ravel()) - network
    voltage = np.zeros(len(nodes))
    i = np.searchsorted(nodes, source)
    voltage[i] = 1.
    if interior.any():
        voltage[interior] = spsolve(laplacian[interior][:, interior].tocsc(),
                                    network[interior, i].toarray().ravel())
    out = {}
    for move in Move:
        q = destination(memory, origin, move)
        if q is None or index(q) not in nodes:
            continue
        j = np.searchsorted(nodes, index(q))
        out[move] = float(network[i, j]) * max(0., 1. - voltage[j])
    return out


class PassageFlow(StreetTrunk):
    def direction_votes(self, view, readings, old, prior, sites):
        parent = super().direction_votes(view, readings, old, prior, sites)
        x, y = view.position
        goal = x + view.dx, y + view.dy
        result = []
        for scores, targets, distance in zip(parent, sites, readings):
            flow = flows(view.memory, view.position, goal, targets, distance)
            weighted = {m: score * flow.get(m, 0.) for m, score in scores.items()}
            result.append(weighted if any(weighted.values()) else scores)
        return result
