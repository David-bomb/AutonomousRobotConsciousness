"""Оценивать выезд к удалённой станции и дальнейший путь на одних и тех же дорогах."""
from bisect import bisect_right
from functools import lru_cache
from itertools import accumulate
from random import Random

import numpy as np
from scipy.sparse.csgraph import connected_components

from .conditional_mission import required_after_stop
from .deadline_goal import goal_deadline
from .frontier_automaton import _arrive
from .frontier_reserve import _lengths
from .grid import locations
from .passage_flow import passage_graph
from .second_window import ReturnWindows


def attach(start, tree, links, rows, cumulative, rng):
    """Добавить к уже построенным дорогам случайный путь после удаления петель."""
    last = {}
    q = start
    while q not in tree:
        choices = cumulative[q]
        index = bisect_right(choices, rng.random() * choices[-1])
        last[q] = links[rows[q] + min(index, len(choices)-1)]
        q = last[q]
    q = start
    while q not in tree:
        tree[q] = last[q]
        q = last[q]
    path = [start]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return path


def connection(a, b, between_roots):
    """Длина пути с общим участком или по раскрытым дорогам между двумя окнами."""
    if a[-1] != b[-1]:
        return len(a) + len(b) - 2 + between_roots
    common = 0
    for p,q in zip(reversed(a),reversed(b)):
        if p != q:
            break
        common += 1
    return len(a) + len(b) - 2*common


@lru_cache(maxsize=32)
def connected_trips(memory, goal, points, origin, gauges):
    width = len(memory[0])
    def index(p):
        return p[0] + width*p[1]
    def point(i):
        return i%width, i//width
    stops = [(p,k) for k in "FR" for p in locations(memory,k)] + [(goal,"B")]
    known = _lengths(memory,origin)
    if goal in known or all(p in known for p,_ in stops[:-1]):
        return ()
    graph = passage_graph(memory,goal)
    rows, links = graph.indptr.tolist(), graph.indices.tolist()
    weights = graph.data.tolist()
    cumulative = [tuple(accumulate(weights[a:b])) for a,b in zip(rows,rows[1:])]
    roots = {index(p) for p in points}
    anchors = roots | {index(p) for p,_ in stops if p in known}
    _, components = connected_components(graph,directed=False)
    reachable = {components[i] for i in roots}
    if any(components[index(p)] not in reachable for p,_ in stops if p not in known):
        return ()
    known_between = {a:_lengths(memory,point(a)) for a in anchors}
    kinds = [(k,i) for i,(_,k) in enumerate(stops)]
    result = []
    rng = Random(42)
    for _ in range(32):
        tree = dict.fromkeys(roots)
        paths = {}
        # Финиш и сервисы присоединяются к одной сети; у ветвей бывает общий участок.
        for p,_ in reversed(stops):
            paths[p] = [index(p)] if p in known else attach(index(p),tree,links,rows,cumulative,rng)
        distances = np.zeros((len(stops),len(stops)))
        candidates = []
        for i,(p,k) in enumerate(stops):
            path = paths[p]
            root = point(path[-1])
            approach = len(path)-1
            if k != "B":
                bound = gauges[int(k == "R")]
                if bound is not None and known[root]+approach < bound:
                    break
                if p not in known:
                    candidates.append((i,root,approach,k))
            for j,(q,_) in enumerate(stops[:i]):
                previous = paths[q]
                bridge = known_between[path[-1]][point(previous[-1])]
                distances[i,j] = distances[j,i] = connection(path,previous,bridge)
        else:
            need = required_after_stop(kinds,distances)
            goal_path = paths[goal]
            result.append((point(goal_path[-1]),len(goal_path)-1,
                           tuple((root,leg,k,need[i]) for i,root,leg,k in candidates)))
    return tuple(result)


class ConnectedExcursions(ReturnWindows):
    def chance(self, view, goal, point, fuel, health, curves):
        direct = super().chance(view,goal,point,fuel,health,curves)
        if goal_deadline(view):
            return direct
        trips = connected_trips(view.memory,goal,tuple(curves),view.position,
                                (view.fuel_distance,view.repair_distance))
        additional = 0
        for end,length,stops in trips:
            if end == point and length <= min(fuel,health):
                continue
            for root,leg,k,need in stops:
                if root != point:
                    continue
                stocks = _arrive(k,leg,fuel,health)
                if stocks and stocks[int(k == "F")] >= need:
                    additional += 1
                    break
        return min(1.,direct + additional/len(trips)) if trips else direct
