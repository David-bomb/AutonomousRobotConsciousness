"""Голосование возможных сетей дорог по двум показаниям из фактической клетки."""
from functools import lru_cache,partial

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

from .frontier_automaton import _paths
from .excursion_automaton import RangeExcursion
from .frontier_reserve import ReserveFrontier
from .grid import destination, locations
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import possible_directions, service_order


@lru_cache(maxsize=32)
def lattice(width, height):
    cells = np.arange(width*height).reshape(height, width)
    edges = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        x = slice(max(0,-dx), min(width,width-dx))
        y = slice(0,height-dy)
        p = cells[y,x].ravel()
        q = p + dx + width*dy
        # Оба угла диагонали должны допускать проезд.
        edges.append((p,q,p+dx,p+width*dy))
    return np.concatenate(edges,axis=1)


def networks(memory,coupled=False,draws=32):
    width, height = len(memory[0]), len(memory)
    n = width*height
    board = np.frombuffer("".join(memory).encode("ascii"),dtype="S1")
    edges = lattice(width,height)
    allowed = (board[edges] != b"#").all(axis=0)
    edges = edges[:,allowed]
    u,v,a,b = edges
    known = (board[edges] != b"?").all(axis=0)
    diag = (a!=u) & (b!=u)
    observed_roads = np.count_nonzero((board!=b"#") & (board!=b"?"))
    known_count = int(known.sum())
    # Открытые циклы повышают правдоподобие дополнительных связей в неизвестной зоне.
    loops = (1+max(0,known_count-observed_roads+1))/(2+known_count)
    diagonal_share = (1+np.count_nonzero(known & diag))/(2+known_count)
    shape = n,n
    rng = np.random.default_rng(42)
    def draw():
        return rng.random(len(allowed))[allowed] if coupled else rng.random(len(u))
    for _ in range(draws):
        # Известные рёбра сохраняются целиком. В остальной области связное
        # дерево задаёт возможные обходы, дополнительные рёбра допускают петли.
        weights = draw()+0.01
        weights[diag & ~known] /= diagonal_share
        weights[known] *= 0.0001
        candidate = coo_matrix((weights,(u,v)), shape=shape).tocsr()
        forest = minimum_spanning_tree(candidate)
        extra = known | (draw() < loops*np.where(diag,diagonal_share,1.))
        yield forest + coo_matrix((np.ones(extra.sum()),(u[extra],v[extra])),shape=shape).tocsr()


@lru_cache(maxsize=96)
def field_votes(memory, origin, readings, old, prior, sites,coupled=False):
    width = len(memory[0])
    index = lambda p: p[1]*width+p[0]
    origin_id = index(origin)
    neighbors = [(m, index(q)) for m in Move if (q:=destination(memory,origin,m)) is not None]
    locations_by_kind = [[index(p) for p in group] for group in sites]
    votes = [{m: 0. for m,_ in neighbors} for _ in range(2)]
    for graph in networks(memory,coupled):
        fields = [dijkstra(graph, directed=False, unweighted=True, indices=ids, min_only=True)
                  if ids else None for ids in locations_by_kind]
        mismatch = 0.
        for field, actual, previous in zip(fields,readings,old):
            if field is None or actual is None:
                continue
            predicted = field[origin_id]
            mismatch += abs(predicted-actual)
            if previous is not None and prior is not None:
                mismatch += abs((field[index(prior)]-predicted)-(previous-actual))
        if not np.isfinite(mismatch):
            continue
        weight = 1/(1+mismatch)**2
        for i, field in enumerate(fields):
            if field is None:
                continue
            descending = [m for m,j in neighbors if field[j] == field[origin_id]-1]
            for m in descending:
                votes[i][m] += weight/len(descending)
    return votes


class RoadHypotheses(ReserveFrontier):
    vote_fields = staticmethod(field_votes)
    rank_return = False

    def direction_votes(self,view,readings,old,prior,sites):
        return self.vote_fields(view.memory,view.position,readings,old,prior,sites)

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        score = self.direction_votes(view,readings,old,prior,sites)[kind]
        return max(options,key=lambda m:(score[m],m==default))

    def _fallback_move(self, view, goal, stops):
        prior_lengths = self._fallback.last_lengths.copy()
        returning = self._fallback.returning
        service = service_order(view, self._fallback.mode, self.RESERVE)
        default = super()._fallback_move(view, goal, stops)
        return self.refine_approach(view,service,prior_lengths,returning,default)

    def refine_approach(self,view,service,prior_lengths,returning,default):
        if service is None or (returning and not self.rank_return) or "несовместимого" in self.activity:
            return default
        previous = prior_lengths.get(service.mode) if view.moved else None
        if previous is not None and previous < service.distance:
            return default
        options = possible_directions(view, service, previous)
        roads = _paths(view.memory, view.position)
        if default not in options or len(options)<2 or any(
                roads.get(p,(None,))[0] == service.distance for p in service.sites):
            return default
        prior = None
        if view.moved:
            dx,dy = view.last_move.value
            prior = view.position[0]-dx,view.position[1]-dy
        old = tuple(prior_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        sites = tuple(tuple(locations(view.memory,k)) for k in "FR")
        chosen = self.rank_directions(view,(view.fuel_distance,view.repair_distance),old,prior,
                                      sites,options,int(service.mode==Mode.REPAIR),default)
        if chosen != default:
            self.activity = "Согласование двух замеров — " + service.mode.value
        return chosen


class RangeRoads(RangeExcursion, RoadHypotheses):
    """До открытия базы сверяет направления, затем планирует запас для выезда."""


class CoupledRoads(RangeRoads):
    """Новая клетка меняет только ограничения и оценки уже взятых гипотез."""
    vote_fields = staticmethod(partial(field_votes,coupled=True))


class ReturnEvidence(CoupledRoads):
    """После неудачной пробы оценивает оставшиеся выходы из той же клетки."""
    rank_return = True
