"""Из каких достижимых окон открытая область вероятнее соединяется с финишем."""
from functools import lru_cache
from dataclasses import replace

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from .first_window import MarginWindow, first_events
from .frontier_automaton import _arrive, _frontiers, _paths
from .grid import cell, destination, locations
from .mission_order import MissionOrderReserve
from .passage_flow import passage_graph
from .remote_station import can_rejoin


@lru_cache(maxsize=32)
def exit_probabilities(memory, goal, points):
    """Доля обратных путей от цели с первым попаданием в каждое окно за 0–100 ходов."""
    width = len(memory[0])
    index = lambda p: p[0] + p[1] * width
    graph = passage_graph(memory, goal)
    totals = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)
    transition = (graph @ diags(inverse)).tocsr()
    ids = np.array([index(p) for p in points])
    mass = np.zeros(len(totals))
    mass[index(goal)] = 1.
    hits = np.zeros((101, len(ids)))
    for step in range(len(hits)):
        hits[step] = mass[ids]
        mass[ids] = 0.
        if not mass.any():
            break
        mass = transition @ mass
    return dict(zip(points, hits.cumsum(axis=0).T))


@lru_cache(maxsize=32)
def corridor_probabilities(memory, goal, points):
    """Пути в неизвестном коридоре разветвляются без немедленного разворота."""
    graph = passage_graph(memory, goal)
    size = graph.shape[0]
    width = len(memory[0])
    source = goal[0] + width * goal[1]
    ids = np.array([x + width * y for x, y in points])
    edges = graph.tocoo()
    start, end, weights = edges.row, edges.col, edges.data
    keys = start.astype(np.int64) * size + end
    order = np.argsort(keys)
    reverse = order[np.searchsorted(keys[order], end.astype(np.int64) * size + start)]
    degree = np.bincount(start, weights, minlength=size)
    onward = degree[end] - weights[reverse]
    # Единственный выход обратно исключается даже при округлении суммы весов.
    forward = np.bincount(start, minlength=size)[end] > 1
    stops = np.isin(end, ids)
    hits = np.zeros((101, len(ids)))
    if source in ids:
        hits[0, ids == source] = 1.
    elif degree[source] > 0:
        mass = weights * (start == source) / degree[source]
        for step in range(1, len(hits)):
            hits[step] = np.bincount(end, mass, minlength=size)[ids]
            mass[stops] = 0.
            share = np.divide(mass, onward, out=np.zeros_like(mass), where=forward)
            incoming = np.bincount(end, share, minlength=size)
            mass = weights * np.maximum(0., incoming[start] - share[reverse])
            if not mass.any():
                break
    return dict(zip(points, hits.cumsum(axis=0).T))


class ExitFlow(MarginWindow):
    probabilities = staticmethod(exit_probabilities)

    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        offered = {a.point for a in choices}
        if len(offered) < 2 or goal in offered:
            return super().pick(view, goal, choices)
        # Участки вне текущего запаса также поглощают пути. Пока новое окно
        # не достигнуто, границы изученного района сохраняют прежний смысл.
        points = tuple(sorted(p for p, _, _ in first_events(view.memory, view.position,
                                                             frozenset(_frontiers(view.memory)))))
        probabilities = self.probabilities(view.memory, goal, points)
        scores = [probabilities[a.point][min(a.fuel, a.health)] for a in choices]
        best = max(scores)
        return super().pick(view, goal, [a for a, score in zip(choices, scores) if score == best])


class CorridorFlow(ExitFlow):
    probabilities = staticmethod(corridor_probabilities)


class MissionExitFlow(MissionOrderReserve, ExitFlow):
    """До освоения сервисов выбирает порядок заездов, после него оценивает выходы к цели."""


class MissionCorridorFlow(MissionExitFlow):
    probabilities = staticmethod(corridor_probabilities)


@lru_cache(maxsize=32)
def eventual_exits(memory, goal, points, graph_builder=passage_graph):
    """Доля путей, достигающих окна первыми, и длина кратчайшего такого пути."""
    graph = graph_builder(memory, goal)
    width = len(memory[0])
    source = goal[0] + width * goal[1]
    ids = np.array([x + width * y for x, y in points])
    available = np.ones(graph.shape[0], dtype=bool)
    available[ids] = False
    result = np.zeros((len(ids), 101))
    if not available[source]:
        result[ids == source] = 1.
        return dict(zip(points, result))
    inside = np.flatnonzero(available)
    distances = dijkstra(graph[inside][:, inside], indices=np.searchsorted(inside, source),
                         directed=False, unweighted=True)
    reachable = np.isfinite(distances)
    inside, distances = inside[reachable], distances[reachable]
    links = graph[inside][:, ids].tocsc()
    if links.nnz:
        # Подать единицу потока в финиш. Окна поглощают её в пропорции,
        # задаваемой всеми неизвестными путями между финишем и границей.
        degree = np.asarray(graph[inside].sum(axis=1)).ravel()
        system = diags(degree) - graph[inside][:, inside]
        supply = np.zeros(len(inside))
        supply[np.searchsorted(inside, source)] = 1.
        potential = spsolve(system.tocsc(), supply)
        shares = np.maximum(0., links.T @ potential)
        for i, share in enumerate(shares):
            neighbors = links.indices[links.indptr[i]:links.indptr[i+1]]
            if len(neighbors):
                first = int(distances[neighbors].min()) + 1
                result[i, first:] = share
    return dict(zip(points, result))


class ExitCurrent(ExitFlow):
    probabilities = staticmethod(eventual_exits)


class BoundaryFlow(ExitFlow):
    """Выбирает новое окно при утрате подтверждённого возврата и ведёт до первого наблюдения."""
    pending = None
    released = False

    def choose(self, view):
        if view.last_move is None:
            self.pending, self.released = None, False
        if self.pending is not None:
            if sum(row.count("?") for row in view.memory) == self.pending:
                goal = view.position[0] + view.dx, view.position[1] + view.dy
                move = super()._reconsider_sortie(view, goal, (), None)
                if move is not None:
                    return move
            self.pending, self.released = None, True
        if self.released and cell(view.memory, *view.position) in "FR":
            stops = tuple((p, k) for k in "FR" for p in locations(view.memory, k))
            if can_rejoin(view, stops, _paths(view.memory, view.position)):
                self.released = False
        return super().choose(view)

    def _reconsider_sortie(self, view, goal, stops, planned):
        if self.released or not can_rejoin(view, stops, _paths(view.memory, view.position)):
            return None
        q = destination(view.memory, view.position, planned)
        if q is None or q == goal:
            return None
        stock = _arrive(cell(view.memory, *q), 1, view.fuel, view.health)
        if stock and can_rejoin(replace(view, position=q, fuel=stock[0], health=stock[1]),
                                stops, _paths(view.memory, q)):
            return None
        move = super()._reconsider_sortie(view, goal, stops, planned)
        if move is not None:
            self.pending = sum(row.count("?") for row in view.memory)
        return move
