"""План на два хода с учётом сведений, которые может дать первое окно."""
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .grid import destination
from .interface import Move
from .pattern_roads import MarkedStreets, patterns, street_graph
from .road_hypotheses import lattice, networks


@lru_cache(maxsize=96)
def preview(view,readings,old,prior,sites):
    memory=view.memory
    h,w=len(memory),len(memory[0])
    idx=lambda p:p[1]*w+p[0]
    start=idx(view.position)
    x,y=view.position
    goal=x+view.dx,y+view.dy
    board,models,weights=patterns(memory,goal)
    first={m:idx(q) for m in Move if (q:=destination(memory,view.position,m))}
    by_kind=[[idx(p) for p in group] for group in sites]
    edges=lattice(w,h)
    rng=np.random.default_rng(42)
    generic=iter(networks(memory,True,64))
    # Возможные следующие измерения делят гипотезы на группы. Внутри
    # группы второй ход должен быть одним и тем же для всех её вариантов.
    groups={m:defaultdict(lambda:np.zeros((2,len(Move)))) for m in first}
    ample=[stock>=max(abs(view.dx),abs(view.dy)) for stock in (view.fuel,view.health)]
    for _ in range(64):
        model=models[rng.choice(len(models),p=weights)]
        graph=(next(generic) if model is None else street_graph(memory,model,board,edges,rng,idx(goal)))
        graph=(graph+graph.T).tocsr()
        fields=[dijkstra(graph,directed=False,unweighted=True,indices=ids,min_only=True)
                if ids and actual is not None else None for ids,actual in zip(by_kind,readings)]
        error=sum(abs(f[start]-d) for f,d in zip(fields,readings) if f is not None and d is not None)
        if not np.isfinite(error):
            continue
        gradient=sum(abs(f[idx(prior)]-f[start]-(a-d)) for f,a,d in zip(fields,old,readings)
                     if prior is not None and None not in (a,d) and f is not None)
        weight=1/((1+error)*(1+gradient))**2
        for move,q in first.items():
            possible=graph.indices[graph.indptr[q]:graph.indptr[q+1]]
            turns=tuple((i,q+dy*w+dx) for i,(dx,dy) in enumerate(m.value for m in Move)
                        if 0<=q%w+dx<w and 0<=q//w+dy<h and q+dy*w+dx in possible)
            measured=tuple(int(f[q]-f[start]) if f is not None else None for f in fields)
            sensor=(measured,tuple(i for i,_ in turns))
            scores=groups[move][sensor]
            for kind,(f,d,stock) in enumerate(zip(fields,readings,(view.fuel,view.health))):
                if not d or f is None:
                    continue
                for i,end in turns:
                    extra=0 if q in by_kind[kind] else int(2+f[end]-f[start])
                    if ample[kind] or extra<=stock-d:
                        scores[kind,i]+=weight/(1+extra)
    return [{m:sum(score[kind].max() for score in partition.values()) for m,partition in groups.items()}
            for kind in range(2)]


class PreviewRoads(MarkedStreets):
    direction_votes=staticmethod(preview)
