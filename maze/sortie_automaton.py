"""Выбор последней поездки после освоения возобновляемой пары станций."""
from functools import lru_cache

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .excursion_automaton import lookouts
from .frontier_automaton import _frontiers
from .frontier_reserve import _estimated_lengths, _lengths, _optimistic
from .grid import destination
from .interface import Move
from .pattern_roads import PatternRoads, patterns, street_graph
from .road_hypotheses import lattice, networks
from .frontier_automaton import _paths


@lru_cache(maxsize=128)
def side_views(memory,goal):
    outward=lookouts(memory,goal)
    distances=_lengths(_optimistic(memory),goal)
    return {p:distances[p] for p in _frontiers(memory) if p not in outward and p in distances}


@lru_cache(maxsize=128)
def scouting_lengths(memory,goal):
    projected=_optimistic(memory)
    remaining=_estimated_lengths(memory,(goal,))
    estimates={}
    for p in lookouts(memory,goal):
        choices=[]
        for move in Move:
            q=destination(projected,p,move)
            if q in remaining and destination(memory,p,move) is None:
                choices.append(2+remaining[q])
        if choices:
            estimates[p]=min(choices)
    return estimates


class SurveyStreets(PatternRoads):
    """До опасного выезда использует доступные из безопасности боковые окна."""
    extra_lookouts=staticmethod(side_views)


class ScoutingSortie(PatternRoads):
    """За рубежом возврата закладывает на неизвестный участок один разведочный ход."""
    onward_estimate=staticmethod(scouting_lengths)


@lru_cache(maxsize=96)
def outside_samples(memory,goal):
    h,w = len(memory),len(memory[0])
    index = lambda p:p[1]*w+p[0]
    board,models,weights = patterns(memory,goal)
    stops = lookouts(memory,goal)
    edges = lattice(w,h)
    projected = _optimistic(memory)
    # Область точки наблюдения включает только дороги, доступные по памяти.
    # Путь после окончательного выезда не должен возвращаться в эту область.
    components = {}
    for point in stops:
        if point in components:
            continue
        component = frozenset(_paths(memory,point))
        components.update(dict.fromkeys(component,component))
    groups = {}
    for point in stops:
        component = components[point]
        if goal in component:
            continue
        ways = [index(q) for m in Move if (q:=destination(projected,point,m)) is not None
                and q not in component]
        if ways:
            groups.setdefault(component,{})[point]=ways
    rng = np.random.default_rng(42)
    fallback = iter(networks(memory,True,24))
    total = {}
    for _ in range(24):
        model = models[rng.choice(len(models),p=weights)]
        graph = (next(fallback) if model is None else
                 street_graph(memory,model,board,edges,rng,index(goal)))
        graph = (graph+graph.T).tocsr()
        for component,gateways in groups.items():
            exterior = np.ones(w*h,dtype=bool)
            exterior[[index(p) for p in component]] = False
            ids = np.flatnonzero(exterior)
            field = np.full(w*h,np.inf)
            field[ids] = dijkstra(graph[ids][:,ids],directed=False,unweighted=True,
                                  indices=np.searchsorted(ids,index(goal)))
            for p,ways in gateways.items():
                i=index(p)
                available=graph.indices[graph.indptr[i]:graph.indptr[i+1]]
                distance=min((1+field[q] for q in ways if q in available),default=np.inf)
                # 100 — вместимость обоих запасов для беспосадочной поездки.
                total[p]=total.get(p,0.)+min(distance,101)
    return {p:s/24 for p,s in total.items()}


class TerminalStreets(PatternRoads):
    """Перед необратимым выездом оценивает длину продолжения по гипотезам о дорогах."""
    onward_estimate=staticmethod(outside_samples)
