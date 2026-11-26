"""Выбор очередности обратимых проб на дороге к одному сервису."""
from functools import lru_cache
from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import dijkstra

from .fork_budget import fork_counts
from .grid import cell,destination
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions
from .road_hypotheses import RangeRoads,networks


@lru_cache(maxsize=2048)
def enough(trials,forks):
    if trials<0:
        return 0.
    if trials>=forks:
        return 1.
    return sum(comb(forks,i) for i in range(trials+1))/2**forks


@lru_cache(maxsize=96)
def ordered_votes(view,readings,old,prior,sites):
    grid=view.memory
    h,w=len(grid),len(grid[0])
    index=lambda p:p[1]*w+p[0]
    center=index(view.position)
    neighbors={m:index(q) for m in Move if (q:=destination(grid,view.position,m))}
    stocks=(view.fuel,view.health)
    by_kind=[[index(p) for p in group] for group in sites]
    choices=[]
    for i,mode in enumerate((Mode.FUEL,Mode.REPAIR)):
        d=readings[i]
        options=possible_directions(view,Service(mode,stocks[i],d,sites[i]),old[i]) if d else {}
        choices.append([m for m in Move if m in options])
    rewards=[np.zeros((len(moves),len(moves))) for moves in choices]
    popular=[np.zeros(len(moves)) for moves in choices]
    incomplete=np.array([any(cell(grid,x+dx,y+dy)=="?" for dx,dy in (m.value for m in Move))
                         for y in range(h) for x in range(w)])
    for graph in networks(grid):
        paths=[dijkstra(graph,directed=False,unweighted=True,indices=ids,min_only=True,return_predecessors=True)
               if ids else None for ids in by_kind]
        error=sum(abs(triple[0][center]-d) for triple,d in zip(paths,readings) if triple is not None and d is not None)
        if not np.isfinite(error):
            continue
        if prior is not None:
            error+=sum(abs(triple[0][index(prior)]-triple[0][center]-(prev-d))
                       for triple,d,prev in zip(paths,readings,old) if triple is not None and prev is not None and d is not None)
        weight=1/(1+error)**2
        forks=incomplete & (np.diff((graph+graph.T).tocsr().indptr)>2)
        for i,moves in enumerate(choices):
            if not moves:
                continue
            field,pred,source=paths[i]
            true=[k for k,m in enumerate(moves) if field[neighbors[m]]==field[center]-1]
            trials=(min(stocks[i],stocks[1-i]-1)-readings[i])//2
            for k in true:
                q=neighbors[moves[k]]
                remaining=fork_counts(pred,int(source[q]),forks)(q)
                vote=weight/len(true)
                popular[i][k]+=vote
                for order in range(len(moves)):
                    rewards[i][k,order]+=vote*enough(trials-order,remaining)
    results=[]
    for moves,matrix,posterior in zip(choices,rewards,popular):
        score=dict.fromkeys(neighbors,(0.,0.))
        for k,m in enumerate(moves):
            # Если эту дорогу проверить первой, остальные пробуем в лучшем
            # для них порядке. При равенстве берём более вероятный спуск.
            others=np.delete(matrix[:,1:],k,axis=0)
            i,j=linear_sum_assignment(others,maximize=True)
            score[m]=(matrix[k,0]+others[i,j].sum(),posterior[k])
        results.append(score)
    return results


class ProbeOrder(RangeRoads):
    rank_return=True
    direction_votes=staticmethod(ordered_votes)
