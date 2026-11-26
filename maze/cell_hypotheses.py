"""Проверка гипотез о проходимости клеток обоими фактическими замерами."""
from functools import lru_cache
from math import lgamma

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .grid import destination
from .interface import Move
from .road_hypotheses import RangeRoads, lattice


@lru_cache(maxsize=96)
def occupancy(memory):
    """Вывести частоту дорог из наблюдений с усреднением возможных периодов."""
    board = np.frombuffer("".join(memory).encode("ascii"),dtype="S1")
    height,width = len(memory),len(memory[0])
    y,x = np.indices((height,width)).reshape(2,-1)
    # Отдельно выданные координаты сервисов известны проезжими, но не
    # являются случайной выборкой дорожного покрытия для оценки частот.
    observed = (board==b".") | (board==b"#") | (board==b"B")
    free = observed & (board!=b"#")
    models = []
    for px in range(1,min(3,width)+1):
        for py in range(1,min(3,height)+1):
            bins = x%px + px*(y%py)
            yes = np.bincount(bins[free],minlength=px*py)
            total = np.bincount(bins[observed],minlength=px*py)
            # Для каждого класса клеток интегрируем неизвестную долю дорог
            # с равномерным начальным распределением от нуля до единицы.
            score = sum(lgamma(a+1)+lgamma(n-a+1)-lgamma(n+2) for a,n in zip(yes,total))
            models.append((score,(1+yes[bins])/(2+total[bins])))
    scores = np.array([s for s,_ in models])
    weights = np.exp(scores-max(scores))
    return board,sum(w*p for w,(_,p) in zip(weights,models))/weights.sum()


@lru_cache(maxsize=96)
def cell_votes(memory,origin,readings,old,prior,sites,ranker=None):
    board,prob = occupancy(memory)
    width,height = len(memory[0]),len(memory)
    index = lambda p:p[1]*width+p[0]
    center = index(origin)
    previous = None if prior is None else index(prior)
    neighbors = [(m,index(q)) for m in Move if (q:=destination(memory,origin,m))]
    by_kind = [[index(p) for p in group] for group in sites]
    edges = lattice(width,height)
    edges = edges[:,(board[edges]!=b"#").all(axis=0)]
    n = len(board)
    rng = np.random.default_rng(42)
    votes = [{m:0. for m,_ in neighbors} for _ in range(2)]
    for _ in range(16):
        if ranker:
            rank = ranker(memory,board,prob,rng)
        else:
            sample = rng.random(n)
            rank = (sample/(1-sample))*((1-prob)/prob)
        thresholds = np.unique(np.r_[0.,rank[board==b"?"]])
        rank[board!=b"?"] = 0
        opens = rank[edges].max(axis=0)
        lo,hi = 0,len(thresholds)-1
        best = None
        while lo<=hi:
            mid = (lo+hi)//2
            u,v = edges[:2,opens<=thresholds[mid]]
            graph = coo_matrix((np.ones(len(u)),(u,v)),shape=(n,n)).tocsr()
            fields = [dijkstra(graph,directed=False,unweighted=True,indices=ids,min_only=True)
                      if ids else None for ids in by_kind]
            error = [field[center]-actual for field,actual in zip(fields,readings)
                     if field is not None and actual is not None]
            if not error:
                break
            if np.isfinite(error).all():
                gradient = sum(abs(field[previous]-field[center]-(a-b))
                               for field,a,b in zip(fields,old,readings)
                               if previous is not None and field is not None and a is not None and b is not None)
                penalty = (1+sum(abs(e) for e in error))*(1+gradient)
                if best is None or penalty<best[0]:
                    best = penalty,fields
                if penalty==1:
                    break
            if max(error)>-min(error):
                lo = mid+1
            else:
                hi = mid-1
        if best is None:
            continue
        penalty,fields = best
        for i,field in enumerate(fields):
            if field is None:
                continue
            descend = [m for m,j in neighbors if field[j]==field[center]-1]
            for m in descend:
                votes[i][m] += 1/(penalty**2*len(descend))
    return votes


class CellRoads(RangeRoads):
    vote_fields = staticmethod(cell_votes)
