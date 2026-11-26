"""Гипотезы о широких помещениях, разделённых стенами с проходами."""
from functools import lru_cache
from math import log

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import expit

from .cell_hypotheses import occupancy
from .grid import destination
from .interface import Move
from .pattern_roads import pattern_votes
from .road_hypotheses import lattice
from .street_trunk import StreetTrunk


def separator_prior(board):
    wall=board==b"#"
    free=(board!=b"?") & ~wall
    known=wall|free
    density=(wall.sum()+1)/(known.sum()+2)
    # Соседние стены вдоль строки — наблюдение о продолжении той же стены.
    pairs=wall[:,:-1] & known[:,1:]
    reversed_pairs=wall[:,1:] & known[:,:-1]
    continuity=(2*(wall[:,:-1]&wall[:,1:]).sum()+1)/(pairs.sum()+reversed_pairs.sum()+2)
    continuity=max(density,continuity)
    ends=(free[:-1]&wall[1:]).sum()+(free[1:]&wall[:-1]).sum()
    trials=(free[:-1]&known[1:]).sum()+(free[1:]&known[:-1]).sum()
    rate=(ends+1)/(trials+2)
    odds=(log(rate/(1-rate))+wall.sum(axis=1)*log(continuity/density)
          +free.sum(axis=1)*log((1-continuity)/(1-density)))
    return expit(odds)


@lru_cache(maxsize=128)
def room_predicts_memory(memory):
    """Проверить по известным клеткам, лучше ли их описывают стены помещений, чем периодическая сетка."""
    board = np.asarray([list(row) for row in memory], dtype="S1")
    road = (board != b"#") & (board != b"?")
    if not (road[:-1, :-1] & road[1:, :-1] & road[:-1, 1:] & road[1:, 1:]).any():
        return False
    # Маркеры сервисов даны отдельно. Отложить можно только обычные клетки,
    # тип которых был открыт наблюдением робота.
    seen = (board == b".") | (board == b"#")
    y, x = np.indices(board.shape)
    room_error = street_error = 0.
    # Каждый рисунок с периодом 2 или 3 представлен в нескольких частях.
    # Отложенный цвет не совпадает с фазой проверяемой периодической сетки.
    for k in range(5):
        hidden = seen & ((x + 2*y) % 5 == k)
        if not hidden.any():
            continue
        masked = board.copy()
        masked[hidden] = b"?"
        partial = tuple(row.tobytes().decode() for row in masked)
        _, street = occupancy(partial)
        row, col = separator_prior(masked), separator_prior(masked.T)
        divider = 1 - (1 - row[:, None]) * (1 - col[None, :])
        known = seen & ~hidden
        def road_share(weights):
            return (1 + weights[road & known].sum()) / (2 + weights[known].sum())
        room = divider * road_share(divider) + (1-divider) * road_share(1-divider)
        room_error += ((room[hidden] - road[hidden]) ** 2).sum()
        street_error += ((street.reshape(board.shape)[hidden] - road[hidden]) ** 2).sum()
    return room_error < street_error


def separators(chances,rng):
    selected=set()
    for i in rng.permutation(np.flatnonzero(rng.random(len(chances))<chances)):
        if 0<i<len(chances)-1 and i-1 not in selected and i+1 not in selected:
            selected.add(i)
    return [-1,*sorted(selected),len(chances)]


def room_partition(board,goal,priors,rng):
    h,w=board.shape
    ys,xs=(separators(chance,rng) for chance in priors)
    opened=np.ones((h,w),dtype=bool)
    opened[ys[1:-1],:]=False
    opened[:,xs[1:-1]]=False
    known=(board!=b"?") & (board!=b"#")
    known=known.copy()
    known[goal[1],goal[0]]=True
    wall=board==b"#"
    xrooms=len(xs)-1
    links=[]
    def gate(a,b,cells):
        possible=[p for p in cells if not wall[p]]
        if possible:
            forced=any(known[p] for p in possible)
            links.append((not forced,rng.random(),a,b,possible))
    for j,(top,bottom) in enumerate(zip(ys,ys[1:])):
        for i,(left,right) in enumerate(zip(xs,xs[1:])):
            a=j*xrooms+i
            if right<w:
                gate(a,a+1,[(y,right) for y in range(top+1,bottom)
                             if not wall[y,right-1] and not wall[y,right+1]])
            if bottom<h:
                gate(a,a+xrooms,[(bottom,x) for x in range(left+1,right)
                                  if not wall[bottom-1,x] and not wall[bottom+1,x]])
    return opened, known, wall, sorted(links), xrooms*(len(ys)-1)


def room_connections(links, count):
    parent=list(range(count))
    def root(site):
        while parent[site]!=site:
            parent[site]=parent[parent[site]]
            site=parent[site]
        return site
    for optional,_,a,b,choices in links:
        u,v=root(a),root(b)
        needed = u!=v or not optional
        if needed:
            parent[u]=v
        yield needed, optional, choices


def room_graph(board,goal,edges,priors,rng):
    opened, known, wall, links, count = room_partition(board,goal,priors,rng)
    h,w=board.shape
    extra=rng.random()
    for needed, optional, choices in room_connections(links,count):
        if needed or rng.random()<extra:
            if optional:
                opened[choices[int(rng.integers(len(choices)))]]=True
    opened=(opened|known)&~wall
    a,b=edges[:2,opened.ravel()[edges].all(axis=0)]
    return coo_matrix((np.ones(len(a)),(a,b)),shape=(h*w,h*w)).tocsr()


@lru_cache(maxsize=96)
def room_votes(view,readings,old,prior,sites,exact):
    h,w=len(view.memory),len(view.memory[0])
    board=np.frombuffer("".join(view.memory).encode(),dtype="S1").reshape(h,w)
    open_cell=(board!=b"#") & (board!=b"?")
    if not (open_cell[:-1,:-1]&open_cell[1:,:-1]&open_cell[:-1,1:]&open_cell[1:,1:]).any():
        return None
    x,y=view.position
    goal=x+view.dx,y+view.dy
    priors=separator_prior(board),separator_prior(board.T)
    edges=lattice(w,h)
    center=y*w+x
    index=lambda p:p[1]*w+p[0]
    options={m:index(q) for m in Move if (q:=destination(view.memory,view.position,m))}
    ids=[[index(p) for p in row] for row in sites]
    rng=np.random.default_rng(42)
    votes=[dict.fromkeys(options,0.) for _ in sites]
    for _ in range(64):
        graph=room_graph(board,goal,edges,priors,rng)
        fields=[dijkstra(graph,directed=False,unweighted=True,indices=g,min_only=True) if g else None for g in ids]
        error=sum(abs(f[center]-d) for f,d in zip(fields,readings) if f is not None and d is not None)
        if not np.isfinite(error) or (exact and error):
            continue
        delta=sum(abs(f[index(prior)]-f[center]-(o-d)) for f,o,d in zip(fields,old,readings)
                  if prior is not None and f is not None and o is not None and d is not None)
        if exact and delta:
            continue
        weight=1/((1+error)*(1+delta))**2
        for score,f in zip(votes,fields):
            if f is not None:
                moves=[m for m,q in options.items() if f[q]==f[center]-1]
                for m in moves:
                    score[m]+=weight/len(moves)
    return votes if any(any(v.values()) for v in votes) else None


class RoomPassages(StreetTrunk):
    exact=True

    def direction_votes(self,view,readings,old,prior,sites):
        return room_votes(view,readings,old,prior,sites,self.exact) or super().direction_votes(
            view,readings,old,prior,sites)


class WeightedRooms(RoomPassages):
    exact=False


class RoomReserve(RoomPassages):
    """При отсутствии объяснения помещениями проверить запас между станциями."""
    budget=True


class WeightedRoomReserve(WeightedRooms):
    budget=True


class RoomEvidence(StreetTrunk):
    """Вес помещений и улиц зависит от согласия их прогнозов с обоими замерами."""
    budget=True

    def direction_votes(self,view,readings,old,prior,sites):
        base=super().direction_votes(view,readings,old,prior,sites)
        room=room_votes(view,readings,old,prior,sites,False)
        if room is None:
            return base
        x,y=view.position
        streets=pattern_votes(view.memory,view.position,readings,old,prior,sites,
                              (x+view.dx,y+view.dy),64)
        result=[]
        for row,road_support,room_support in zip(base,streets,room):
            total=sum(row.values())
            if not total:
                result.append(room_support)
                continue
            weight=sum(road_support.values())/total
            result.append({m:weight*v+room_support.get(m,0.) for m,v in row.items()})
        return result
