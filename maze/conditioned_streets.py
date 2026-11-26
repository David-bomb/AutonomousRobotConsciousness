"""Уточнение возможных дорог заменой одного неизвестного отрезка в связной сети."""
from functools import lru_cache

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .grid import destination
from .interface import Move
from .pattern_roads import MarkedStreets,districts,graph_from_parts,patterns,road_parts
from .road_hypotheses import lattice,networks


def exchange(memory,model,tree,rng):
    count,a,b=districts(memory,model)
    missing=np.flatnonzero(~tree & (a!=b))
    if not len(missing):
        return None
    add=int(rng.choice(missing))
    links=[[] for _ in range(count)]
    for e in np.flatnonzero(tree & (a!=b)):
        u,v=int(a[e]),int(b[e])
        links[u].append((v,e))
        links[v].append((u,e))
    source,target=int(a[add]),int(b[add])
    came={source:(source,None)}
    pending=[source]
    while pending and target not in came:
        current=pending.pop()
        for q,e in links[current]:
            if q not in came:
                came[q]=(current,e)
                pending.append(q)
    if target not in came:
        return None
    cycle=[]
    p=target
    while p!=source:
        p,e=came[p]
        cycle.append(e)
    candidate=tree.copy()
    candidate[add]=True
    candidate[int(rng.choice(cycle))]=False
    return candidate


@lru_cache(maxsize=96)
def conditioned_votes(view,readings,old,prior,sites):
    memory=view.memory
    h,w=len(memory),len(memory[0])
    x,y=view.position
    goal=(x+view.dx,y+view.dy)
    index=lambda p:p[1]*w+p[0]
    center=index(view.position)
    board,models,weights=patterns(memory,goal)
    edges=lattice(w,h)
    neighbors={m:index(q) for m in Move if (q:=destination(memory,view.position,m))}
    ids=[[index(p) for p in group] for group in sites]
    def score(graph):
        fields=[dijkstra(graph,directed=False,unweighted=True,indices=g,min_only=True) if g else None for g in ids]
        error=sum(abs(f[center]-d) for f,d in zip(fields,readings) if f is not None and d is not None)
        if not np.isfinite(error):
            return 0.,None
        change=sum(abs(f[index(prior)]-f[center]-(a-d)) for f,a,d in zip(fields,old,readings)
                   if prior is not None and f is not None and a is not None and d is not None)
        # Корень обычного веса: половина согласования применяется при отборе
        # сетей, половина после их локального уточнения.
        weight=1/((1+error)*(1+change))
        downhill=[[m for m,q in neighbors.items() if f[q]==f[center]-1] if f is not None else [] for f in fields]
        return weight,downhill
    rng=np.random.default_rng(42)
    generic=iter(networks(memory,True,64))
    particles=[]
    for _ in range(64):
        model=models[rng.choice(len(models),p=weights)]
        tree=extra=None
        if model is None:
            graph=next(generic)
        else:
            tree,extra=road_parts(memory,model,rng)
            graph=graph_from_parts(memory,model,board,edges,tree|extra,index(goal))
        weight,downs=score(graph)
        if weight>0:
            particles.append((model,tree,extra,weight,downs))
    votes=[dict.fromkeys(neighbors,0.) for _ in sites]
    if not particles:
        return votes
    p=np.array([row[3] for row in particles])
    for draw in rng.choice(len(particles),size=64,p=p/p.sum()):
        model,tree,extra,weight,downs=particles[draw]
        if model is not None:
            for _ in range(3):
                next_tree=exchange(memory,model,tree,rng)
                if next_tree is None:
                    continue
                graph=graph_from_parts(memory,model,board,edges,next_tree|extra,index(goal))
                next_weight,next_downs=score(graph)
                if rng.random()*weight<next_weight:
                    tree,weight,downs=next_tree,next_weight,next_downs
        for target,moves in zip(votes,downs):
            for move in moves:
                target[move]+=weight/len(moves)
    return votes


class ConditionedStreets(MarkedStreets):
    direction_votes=staticmethod(conditioned_votes)
