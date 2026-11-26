"""План на три команды с единым решением после одинаковых возможных наблюдений."""
from collections import defaultdict
from functools import lru_cache
from math import inf, isfinite

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .cell_hypotheses import occupancy
from .frontier_automaton import _arrive
from .frontier_reserve import _lengths, _optimistic
from .grid import destination, locations
from .interface import Move
from .pattern_roads import cells_from_parts, patterns, road_parts
from .pilot_automaton import Mode
from .resource_automaton import service_order
from .road_hypotheses import lattice
from .street_trunk import StreetTrunk


def required_after_stop(sites, fields):
    # На станции восстанавливается ровно один запас. Для достижимости
    # финиша достаточно минимального требования к другому запасу.
    need = [fields[-1,q] if fields[-1,q] <= 100 else inf for _,q in sites[:-1]]
    while True:
        changed = False
        for i,(kind,p) in enumerate(sites[:-1]):
            for j,(next_kind,_) in enumerate(sites[:-1]):
                if i == j or not isfinite(need[j]):
                    continue
                d = fields[j,p]
                required = d+need[j] if kind == next_kind else d if 100-d >= need[j] else inf
                if required <= 100 and required < need[i]:
                    changed = True
                    need[i] = required
        if not changed:
            return need


@lru_cache(maxsize=12)
def possible_worlds(memory, goal):
    height, width = len(memory), len(memory[0])
    index = lambda p: p[1]*width+p[0]
    board, models, likelihood = patterns(memory, goal)
    _, probability = occupancy(memory)
    edges = lattice(width,height)
    sites = [(k,index(p)) for k in "FR" for p in locations(memory,k)] + [("B",index(goal))]
    rng = np.random.default_rng(42)
    results = []
    for _ in range(32):
        model = models[rng.choice(len(models),p=likelihood)]
        if model is None:
            # Одна вероятность заполнения для всего рисунка, индивидуальные
            # шансы клетки задаются наблюдаемой периодичностью покрытия.
            odds = rng.beta(2,2)
            adjusted = probability*odds / (probability*odds + (1-probability)*(1-odds))
            opened = (board != b"#") & ((board != b"?") | (rng.random(len(board)) < adjusted))
            opened[index(goal)] = True
        else:
            tree, extra = road_parts(memory, model, rng)
            opened = cells_from_parts(memory,model,board,tree|extra,index(goal))
        u,v = edges[:2,opened[edges].all(axis=0)]
        graph = coo_matrix((np.ones(len(u)),(u,v)),shape=(len(board),len(board))).tocsr()
        fields = dijkstra(graph,directed=False,unweighted=True,indices=[p for _,p in sites])
        results.append((opened,fields,required_after_stop(sites,fields)))
    return sites,results


def contingent_move(view,old,goal,default):
    width,height = len(view.memory[0]),len(view.memory)
    index = lambda p:p[1]*width+p[0]
    origin = index(view.position)
    prior = None
    if view.moved:
        dx,dy = view.last_move.value
        prior = origin-dx-width*dy
    actual = (view.fuel_distance,view.repair_distance)
    sites,worlds = possible_worlds(view.memory,goal)
    groups = [[j for j,(kind,_) in enumerate(sites) if kind==k] for k in "FR"]
    if not all(groups) or None in actual:
        return default
    kind_at = {q:k for k,q in sites}
    gauges,weights = {},{}
    for i,(_,fields,_) in enumerate(worlds):
        nearest = [fields[ids].min(axis=0) for ids in groups]
        error = sum(abs(f[origin]-d) for f,d in zip(nearest,actual))
        if not isfinite(error):
            continue
        delta = sum(abs(f[prior]-f[origin]-(o-d)) for f,d,o in zip(nearest,actual,old)
                    if prior is not None and o is not None)
        if isfinite(delta):
            gauges[i] = nearest
            weights[i] = 1./((1+error)*(1+delta))**2
    if not weights:
        return default
    lower = [_lengths(_optimistic(view.memory),(q%width,q//width)) for _,q in sites]
    moves = tuple(Move)

    @lru_cache(None)
    def leaf(i,q,f,h):
        _,fields,need = worlds[i]
        if fields[-1,q] <= min(f,h):
            return True
        for j,(kind,p) in enumerate(sites[:-1]):
            k = int(kind=="R")
            if p==q:
                if (h if kind=="F" else f) >= need[j]:
                    return True
                continue
            length = max(lower[j].get((q%width,q//width),inf),
                         fields[j,q] + actual[k] - gauges[i][k][origin])
            if length <= min(f,h):
                after = _arrive(kind,int(length),f,h)
                if after and after[1-k] >= need[j]:
                    return True
        return False

    @lru_cache(None)
    def observation(i,q):
        opened = worlds[i][0]
        x,y = q%width,q//width
        def free(a,b):
            return 0 <= a < width and 0 <= b < height and opened[b*width+a]
        window = tuple(bool(free(x+a,y+b)) for b in (-1,0,1) for a in (-1,0,1))
        roads = tuple((n,q+dx+dy*width) for n,(dx,dy) in enumerate(m.value for m in moves)
                      if free(x+dx,y+dy) and (not dx or not dy or (free(x+dx,y) and free(x,y+dy))))
        delta = tuple(int(field[q]-field[origin]) for field in gauges[i])
        return (window,delta),roads

    def after_move(ids,q,f,h,depth):
        kind = kind_at.get(q,".")
        if kind=="B":
            return sum(weights[i] for i in ids)
        stock = _arrive(kind,1,f,h)
        if not stock:
            return 0.
        if depth==0:
            return sum(weights[i] for i in ids if leaf(i,q,*stock))
        groups_by_view = defaultdict(list)
        for i in ids:
            groups_by_view[observation(i,q)[0]].append(i)
        value = 0.
        for group in groups_by_view.values():
            if len(group)==1:
                i=group[0]
                value+=weights[i]*leaf(i,q,*stock)
                continue
            roads = observation(group[0],q)[1]
            value += max((after_move(group,next_q,*stock,depth-1) for _,next_q in roads),default=0.)
        return value

    scores={m:after_move(tuple(weights),index(q),view.fuel,view.health,2)
            for m in moves if (q:=destination(view.memory,view.position,m)) is not None}
    # Одни и те же веса могут складываться в разных порядках.
    best=max(scores.values(),default=0.)
    tied=[m for m,v in scores.items() if best-v <= 1e-12*sum(weights.values())]
    return default if default in tied or not tied else tied[0]


class ConditionalMission(StreetTrunk):
    def _fallback_move(self,view,goal,stops):
        old=tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        demand=service_order(view,self._fallback.mode,self.RESERVE)
        default=super()._fallback_move(view,goal,stops)
        if demand is None or "несовместимого" in self.activity or sum(
                destination(view.memory,view.position,m) is not None for m in Move)<2:
            return default
        move=contingent_move(view,old,goal,default)
        if move!=default:
            self._fallback.returning=False
            self.activity="Общий ход для одинаковых будущих наблюдений"
        return move
