"""Равновероятные деревья улиц с сохранением уже раскрытых соединений."""
from functools import lru_cache
from random import Random

from .pattern_roads import MarkedStreets, districts, pattern_votes


@lru_cache(maxsize=256)
def connections(memory,model):
    size,left,right=districts(memory,model)
    neighbors=[[] for _ in range(size)]
    for edge,(u,v) in enumerate(zip(left.tolist(),right.tolist())):
        if u!=v:
            neighbors[u].append((v,edge))
            neighbors[v].append((u,edge))
    # Наблюдавшаяся стена может разбить предполагаемую систему улиц.
    # В каждой физически отдельной области нужно своё начальное дерево.
    marked=set()
    roots=[]
    for origin in range(size):
        if origin in marked:
            continue
        roots.append(origin)
        marked.add(origin)
        todo=[origin]
        while todo:
            for q,_ in neighbors[todo.pop()]:
                if q not in marked:
                    marked.add(q)
                    todo.append(q)
    return neighbors,roots


def uniform_tree(memory,model,forced,rng):
    neighbors,roots=connections(memory,model)
    wired=set(roots)
    selected=forced.copy()
    random=Random(int(rng.integers(2**32)))
    next_edge=[None]*len(neighbors)
    # В алгоритме Уилсона при повторном посещении клетки заменяется
    # исходящий переход. При добавлении пути петли уже стёрты.
    for origin in range(len(neighbors)):
        p=origin
        while p not in wired:
            next_edge[p]=random.choice(neighbors[p])
            p=next_edge[p][0]
        p=origin
        while p not in wired:
            wired.add(p)
            p,edge=next_edge[p]
            selected[edge]=True
    return selected


class UniformStreets(MarkedStreets):
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        return pattern_votes(view.memory,view.position,readings,old,prior,sites,
                             (x+view.dx,y+view.dy),self.samples,tree_sampler=uniform_tree)
