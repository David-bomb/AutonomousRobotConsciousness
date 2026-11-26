"""Сравнение похожих наблюдений на независимо сгенерированных дорогах."""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .interface import Move
from .road_hypotheses import lattice
from .room_roads import room_graph
from .second_window import ReturnWindows
from .translated_paths import landscape


@lru_cache(maxsize=18)
def reference_roads(draw):
    """Геометрия примера зависит только от номера случайного рисунка."""
    side = 20
    width = 2 * side + 1
    rng = np.random.default_rng(194711 + draw)
    edges = lattice(width, width)
    if draw % 3 == 2:
        board = np.full((width, width), b"?", dtype="S1")
        board[[0, -1], :] = b"#"
        board[:, [0, -1]] = b"#"
        priors = [np.full(width, rng.uniform(.1, .5)) for _ in range(2)]
        graph = room_graph(board, (side, side), edges, priors, rng)
        degree = graph.getnnz(axis=0) + graph.getnnz(axis=1)
        road = (degree > 0).reshape(board.shape)
    else:
        road = landscape(side, 190 + draw)[0].copy()
        if draw % 3 == 1:
            y, x = np.indices(road.shape)
            doors = (x % 2 != y % 2) & (x > 0) & (y > 0) & (x < width-1) & (y < width-1)
            road |= doors & (rng.random(road.shape) < rng.random())
        u, v = edges[:2, road.ravel()[edges].all(axis=0)]
        graph = coo_matrix((np.ones(len(u)), (u, v)), shape=(road.size, road.size)).tocsr()
    # Индекс за правым краем таблицы обозначает отсутствие дороги.
    missing = road.size
    distances = np.full((missing + 1, missing + 1), 101, dtype=np.uint8)
    for sources in np.array_split(np.flatnonzero(road), 8):
        if sources.size:
            fields = dijkstra(graph, directed=False, unweighted=True, indices=sources, limit=100)
            distances[sources, :missing] = np.minimum(fields, 101).astype(np.uint8)
    return road, distances


def shifted(width, x, y, offset):
    xx, yy = x + offset[0], y + offset[1]
    inside = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < width)
    return np.where(inside, yy * width + xx, width*width)


@lru_cache(maxsize=128)
def reference_votes(view, readings, old, prior, sites, options, kind):
    """Сравнить одинаковые окна и реальные замеры в тех же относительных положениях."""
    ox, oy = view.position
    previous = None if prior is None else (prior[0]-ox, prior[1]-oy)
    offsets = [[(x-ox, y-oy) for x, y in group
                if readings[i] is not None and max(abs(x-ox), abs(y-oy)) <= readings[i]+2]
               for i, group in enumerate(sites)]
    context = [(x-ox, y-oy, cell != "#") for y, row in enumerate(view.memory) for x, cell in enumerate(row)
               if cell != "?" and 1 < max(abs(x-ox), abs(y-oy)) <= 3]
    votes = np.zeros(len(options))
    examples = 0
    for draw in range(18):
        road, fields = reference_roads(draw)
        width = len(road)
        y, x = np.where(road)
        for j, row in enumerate(view.window):
            for i, cell in enumerate(row):
                match = road[y+j-1, x+i-1] == (cell != "#")
                y, x = y[match], x[match]
        if not x.size:
            continue
        center = y*width + x
        earlier = shifted(width, x, y, previous) if previous is not None else None
        match = np.ones(x.shape, dtype=bool)
        groups = []
        for i, displacements in enumerate(offsets):
            targets = [shifted(width, x, y, offset) for offset in displacements]
            groups.append(targets)
            if readings[i] is None:
                continue
            if not targets:
                match[:] = False
                break
            # Положение каждого потенциально ближайшего сервиса должно быть дорогой.
            match &= np.logical_and.reduce([fields[t, t] == 0 for t in targets])
            match &= np.minimum.reduce([fields[center, t] for t in targets]) == readings[i]
            if earlier is not None and old[i] is not None:
                match &= np.minimum.reduce([fields[earlier, t] for t in targets]) == old[i]
        if not match.any():
            continue
        y, x = y[match], x[match]
        targets = [t[match] for t in groups[kind]]
        slopes = []
        for move in options:
            q = shifted(width, x, y, move.value)
            slopes.append(np.minimum.reduce([fields[q, t] for t in targets]) == readings[kind]-1)
        slopes = np.asarray(slopes)
        if not slopes.any():
            continue
        misses = np.zeros(len(x), dtype=int)
        for dx, dy, free in context:
            p = shifted(width, x, y, (dx, dy))
            misses += (fields[p, p] == 0) != free
        weights = 1. / (1 + misses)**2
        votes += (slopes * (weights / np.maximum(1, slopes.sum(axis=0)))).sum(axis=1)
        examples += 1
    return examples, tuple(votes)


class ReferenceRoads(ReturnWindows):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        own, other = (view.fuel, view.health) if kind == 0 else (view.health, view.fuel)
        if readings[kind] is None or not 0 <= min(own, other-1) - readings[kind] < 2:
            return parent
        examples, votes = reference_votes(view, readings, old, prior, sites, tuple(options), kind)
        if examples < 2:
            return parent
        return max(zip(options, votes), key=lambda pair: (pair[1], pair[0] == parent))[0]
