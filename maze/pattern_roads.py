"""Гипотезы связности с шагом улиц, оценённым по уже увиденным клеткам."""
from functools import lru_cache
from math import lgamma,log

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra,minimum_spanning_tree

from .grid import destination
from .interface import Move
from .road_hypotheses import RangeRoads,field_votes,lattice,networks


def beta_score(roads,walls,a,b):
    return (lgamma(a+roads)+lgamma(b+walls)-lgamma(a+b+roads+walls)
            -lgamma(a)-lgamma(b)+lgamma(a+b))


@lru_cache(maxsize=128)
def patterns(memory,marker_goal=None):
    h,w = len(memory),len(memory[0])
    y,x = np.indices((h,w)).reshape(2,-1)
    board = np.frombuffer("".join(memory).encode(),dtype="S1")
    free = (board==b".")|(board==b"B")
    wall = board==b"#"
    choices = [(None,beta_score(free.sum(),wall.sum(),1,1))]
    spacings = [(a,b) for a in (2,3) for b in (2,3) if a<=w and b<=h]
    for sx,sy in spacings:
        for ox in range(sx):
            for oy in range(sy):
                col,row = x%sx==ox,y%sy==oy
                masks = (col & row,col ^ row,~(col|row))
                score = -log(len(spacings)*sx*sy)
                marker_probs = np.zeros(len(board)) if marker_goal is not None else None
                for mask,(a,b) in zip(masks,((3,1),(1,1),(1,3))):
                    nf,nw=(free&mask).sum(),(wall&mask).sum()
                    score += beta_score(nf,nw,a,b)
                    if marker_probs is not None:
                        marker_probs[mask]=(nf+a)/(nf+nw+a+b)
                if marker_probs is not None:
                    markers=(board==b"F") | (board==b"R")
                    gx,gy=marker_goal
                    markers[gy*w+gx]=True
                    # Положение сервиса свидетельствует о наличии дороги в
                    # этой клетке. Корректируем правдоподобие шага и сдвига,
                    # сохраняя оценку доли дорог по реально видимому покрытию.
                    score+=np.log(marker_probs[markers]/marker_probs.mean()).sum()
                choices.append(((sx,sy,ox,oy),score))
    logweights = np.array([s for _,s in choices])
    weights = np.exp(logweights-logweights.max())
    return board,tuple(k for k,_ in choices),weights/weights.sum()


@lru_cache(maxsize=512)
def scaffold(memory,model):
    h,w = len(memory),len(memory[0])
    board = np.frombuffer("".join(memory).encode(),dtype="S1")
    sx,sy,ox,oy = model
    centers = np.arange(w*h).reshape(h,w)[oy::sy,ox::sx]
    edges = []
    for p in centers.ravel():
        if p%w+sx<w:
            edges.append((p,p+sx,np.arange(p+1,p+sx)))
        if p//w+sy<h:
            edges.append((p,p+sy*w,np.arange(p+w,p+sy*w,w)))
    valid = [(u,v,door) for u,v,door in edges
             if board[u]!=b"#" and board[v]!=b"#" and (board[door]!=b"#").all()]
    u = np.array([e[0] for e in valid],dtype=int)
    v = np.array([e[1] for e in valid],dtype=int)
    doors = [e[2] for e in valid]
    forced = np.array([(board[d]!=b"?").all() for d in doors],dtype=bool)
    all_doors = [board[d] for _,_,d in edges]
    open_obs = sum((d!=b"?").all() and (d!=b"#").all() for d in all_doors)
    wall_obs = sum((d==b"#").any() for d in all_doors)
    return centers.ravel(),u,v,doors,forced,open_obs,wall_obs


@lru_cache(maxsize=512)
def districts(memory,model):
    _,u,v,_,forced,_,_=scaffold(memory,model)
    nodes,number=np.unique(np.r_[u,v],return_inverse=True)
    a,b=np.split(number,2)
    n=len(nodes)
    fixed=coo_matrix((np.ones(forced.sum()),(a[forced],b[forced])),shape=(n,n)).tocsr()
    count,labels=connected_components(fixed,directed=False)
    return count,labels[a],labels[b]


def depth_tree(memory,model,forced,rng):
    count,a,b=districts(memory,model)
    links=[[] for _ in range(count)]
    for e in rng.permutation(np.flatnonzero(a!=b)):
        p,q=int(a[e]),int(b[e])
        links[p].append((q,e))
        links[q].append((p,e))
    used=forced.copy()
    seen=set()
    for root in rng.permutation(count):
        if root in seen:
            continue
        seen.add(root)
        stack=[root]
        while stack:
            if not links[stack[-1]]:
                stack.pop()
                continue
            q,e=links[stack[-1]].pop()
            if q not in seen:
                seen.add(q)
                used[e]=True
                stack.append(q)
    return used


def road_parts(memory,model,rng,depth=False,tree_sampler=None):
    centers,u,v,doors,forced,open_obs,wall_obs = scaffold(memory,model)
    n = len(memory)*len(memory[0])
    if tree_sampler is not None:
        used=tree_sampler(memory,model,forced,rng)
    elif depth:
        used=depth_tree(memory,model,forced,rng)
    else:
        weight = rng.random(len(u))+.001
        weight[forced] *= .0001
        tree = minimum_spanning_tree(coo_matrix((weight,(u,v)),shape=(n,n)).tocsr())
        used = np.zeros(len(u),dtype=bool)
        if len(u):
            used |= np.asarray(tree[u,v]).ravel()>0
    # У дерева N-1 связей. При избытке открытых дорог добавляем петли.
    fill = rng.beta(1+open_obs,1+wall_obs)
    already = used.mean() if len(u) else 1.
    extra = max(0.,(fill-already)/(1-already)) if already<1 else 0.
    return used,forced | (rng.random(len(u))<extra)


def cells_from_parts(memory,model,board,used,known_goal):
    centers,_,_,doors,_,_,_ = scaffold(memory,model)
    opened = (board!=b"?") & (board!=b"#")
    opened[centers] = board[centers]!=b"#"
    if known_goal is not None:
        opened[known_goal]=True
    for i in np.flatnonzero(used):
        opened[doors[i]] = True
    return opened


def graph_from_parts(memory,model,board,edges,used,known_goal):
    opened = cells_from_parts(memory,model,board,used,known_goal)
    n=len(board)
    a,b = edges[:2,opened[edges].all(axis=0)]
    return coo_matrix((np.ones(len(a)),(a,b)),shape=(n,n)).tocsr()


def street_graph(memory,model,board,edges,rng,known_goal=None,depth=False,tree_sampler=None):
    tree,extra=road_parts(memory,model,rng,depth,tree_sampler)
    return graph_from_parts(memory,model,board,edges,tree|extra,known_goal)


@lru_cache(maxsize=96)
def pattern_votes(memory,position,readings,old,prior,sites,marker_goal=None,samples=64,mixed=False,
                  rule=None,stocks=None,split_votes=False,tree_sampler=None):
    h,w = len(memory),len(memory[0])
    index = lambda p:p[1]*w+p[0]
    center = index(position)
    neighbors = [(m,index(q)) for m in Move if (q:=destination(memory,position,m))]
    by_kind = [[index(p) for p in group] for group in sites]
    board,models,weights = patterns(memory,marker_goal)
    edges = lattice(w,h)
    rng = np.random.default_rng(42)
    groups = [[dict.fromkeys((m for m,_ in neighbors),0.) for _ in sites]
              for _ in range(2 if split_votes else 1)]
    generic = iter(networks(memory,True,samples if marker_goal is not None else 32))
    for trial in range(samples):
        votes=groups[trial%len(groups)]
        model = models[rng.choice(len(models),p=weights)]
        if model is None:
            graph = next(generic,None)
            if graph is None:
                generic = iter(networks(memory,True))
                graph = next(generic)
        else:
            graph = street_graph(memory,model,board,edges,rng,index(marker_goal) if marker_goal is not None else None,
                                 depth=mixed and rng.random()<0.5,tree_sampler=tree_sampler)
        fields = [dijkstra(graph,directed=False,unweighted=True,indices=ids,min_only=True)
                  if ids else None for ids in by_kind]
        mismatch = sum(abs(field[center]-d) for field,d in zip(fields,readings) if field is not None and d is not None)
        if not np.isfinite(mismatch):
            continue
        gradient = sum(abs((field[index(prior)]-field[center])-(prev-d))
                       for field,d,prev in zip(fields,readings,old)
                       if prior is not None and prev is not None and d is not None and field is not None)
        weight = 1/((1+mismatch)*(1+gradient))**2
        if rule=="common":
            descending=[{m for m,q in neighbors if field is not None and field[q]==field[center]-1}
                        for field in fields]
            shared=descending[0]&descending[1]
            for scores,own in zip(votes,descending):
                choices=shared or own
                for move in choices:
                    scores[move]+=weight/len(choices)
            continue
        if rule:
            for move,q in neighbors:
                # Один ход уже оплачен. Лишние ходы до сервиса относительно
                # кратчайшего пути равны 0, 1 или 2 в данной гипотезе.
                excess = [1+field[q]-field[center] if field is not None else 0 for field in fields]
                for i in range(len(votes)):
                    if fields[i] is None:
                        continue
                    if rule == "all":
                        value = float(excess[i] == 0)
                    else:
                        if any(d is not None and s < max(abs(position[0]-marker_goal[0]),
                               abs(position[1]-marker_goal[1])) and s-d<extra
                               for d,s,extra in zip(readings,stocks,excess)):
                            continue
                        value = max(0.,1-excess[i]/2)
                    votes[i][move] += weight*value
            continue
        for score,field in zip(votes,fields):
            if field is None:
                continue
            moves = [m for m,q in neighbors if field[q]==field[center]-1]
            for m in moves:
                score[m] += weight/len(moves)
    return groups if split_votes else groups[0]


class PatternRoads(RangeRoads):
    vote_fields = staticmethod(pattern_votes)


class MarkedStreets(PatternRoads):
    samples=64
    mixed=False

    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        return pattern_votes(view.memory,view.position,readings,old,prior,sites,
                             (x+view.dx,y+view.dy),self.samples,self.mixed)


class ManyStreets(MarkedStreets):
    """Больше независимых сетей для редкого сочетания двух показаний."""
    samples=256


class MixedStreets(ManyStreets):
    """Проверяет и частые короткие ветви, и длинные проходы с редкими развилками."""
    mixed=True


class AllApproaches(MarkedStreets):
    """Каждая гипотеза целиком поддерживает все сокращающие путь ходы."""
    rule = "all"

    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        return pattern_votes(view.memory,view.position,readings,old,prior,sites,
                             (x+view.dx,y+view.dy),self.samples,self.mixed,
                             self.rule,(view.fuel,view.health))


class InsuredApproaches(AllApproaches):
    """Оценивает также боковой ход и риск потери любого необходимого сервиса."""
    rule = "insured"


class CommonVote(AllApproaches):
    """Если в предполагаемой сети есть ход к обеим станциям, голосует за него."""
    rule = "common"


class StableStreets(PatternRoads):
    """Изменяет родительское направление, если его независимо поддержали две группы сетей."""
    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        halves=pattern_votes(view.memory,view.position,readings,old,prior,sites,split_votes=True)
        ranks=[max(options,key=lambda m:(part[kind][m],m==default)) for part in halves]
        if ranks[0]==ranks[1] and all(part[kind][ranks[0]]>0 for part in halves):
            return ranks[0]
        irregular=field_votes(view.memory,view.position,readings,old,prior,sites)[kind]
        return max(options,key=lambda m:(irregular[m],m==default))
