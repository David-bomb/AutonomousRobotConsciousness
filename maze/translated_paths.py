"""Проверка разных размещений наблюдаемого фрагмента в независимых моделях дорог."""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .interface import Move
from .street_trunk import StreetTrunk,StreetTrunkReserve


@lru_cache(maxsize=32)
def landscape(side, draw):
    rng = np.random.default_rng(1009+draw)
    nodes = np.arange(side*side).reshape(side,side)
    a = np.r_[nodes[:,:-1].ravel(), nodes[:-1].ravel()]
    b = np.r_[nodes[:,1:].ravel(), nodes[1:].ravel()]
    if draw%2:
        graph = coo_matrix((rng.random(len(a))+.001,(a,b)),shape=(nodes.size,nodes.size))
        u,v = minimum_spanning_tree(graph.tocsr()).nonzero()
    else:
        edges = [[] for _ in range(nodes.size)]
        for e in rng.permutation(len(a)):
            i,j = a[e],b[e]
            edges[i].append(j)
            edges[j].append(i)
        stack = [int(rng.integers(nodes.size))]
        seen = {stack[0]}
        tree = []
        while stack:
            p = stack[-1]
            if not edges[p]:
                stack.pop()
                continue
            q = edges[p].pop()
            if q not in seen:
                seen.add(q)
                tree.append((p,q))
                stack.append(q)
        u,v = np.array(tree).T
    width = 2*side+1
    road = np.zeros((width,width),bool)
    road[1::2,1::2] = True
    road[1+u//side+v//side, 1+u%side+v%side] = True
    terrain = road.ravel()
    ids = np.arange(terrain.size, dtype=np.int32)
    parent = ids.copy()
    depth = np.zeros_like(ids)
    start = width+1
    todo = [start]
    while todo:
        i = todo.pop()
        for q in (i-1,i+1,i-width,i+width):
            if terrain[q] and q!=parent[i]:
                parent[q]=i
                depth[q]=depth[i]+1
                todo.append(q)
    x,y = ids%width,ids//width
    jump = [parent]
    box = [np.array((np.minimum(x,x[parent]),np.minimum(y,y[parent]),
                     np.maximum(x,x[parent]),np.maximum(y,y[parent])),dtype=np.int16)]
    while 1<<len(jump) <= depth.max():
        j=jump[-1]
        low=np.minimum(box[-1][:2],box[-1][:2,j])
        high=np.maximum(box[-1][2:],box[-1][2:,j])
        box.append(np.vstack((low,high)))
        jump.append(j[j])
    return road,depth,jump,box


def climb(nodes,steps,jump,boxes=None,crop=None):
    pos=nodes.copy()
    valid=np.ones(len(pos),bool)
    for bit,anc in enumerate(jump):
        change=(steps & (1<<bit))!=0
        if not change.any():
            continue
        selected=pos[change]
        if boxes is not None:
            x0,y0,x1,y1=crop
            low_x,low_y,high_x,high_y=boxes[bit][:,selected]
            valid[change]&=(low_x>=x0[change]) & (low_y>=y0[change]) & (high_x<x1[change]) & (high_y<y1[change])
        pos[change]=anc[selected]
    return pos,valid


def routes_to(source,target,depth,jump,boxes,crop):
    a,_=climb(source,np.maximum(0,depth[source]-depth[target]),jump)
    b,_=climb(target,np.maximum(0,depth[target]-depth[source]),jump)
    for ancestors in reversed(jump):
        different=ancestors[a]!=ancestors[b]
        a[different],b[different]=ancestors[a[different]],ancestors[b[different]]
    common=np.where(a==b,a,jump[0][a])
    up,down=depth[source]-depth[common],depth[target]-depth[common]
    _,left=climb(source,up,jump,boxes,crop)
    _,right=climb(target,down,jump,boxes,crop)
    next_on_branch,_=climb(target,np.maximum(0,down-1),jump)
    first=np.where(up>0,jump[0][source],next_on_branch)-source
    return np.where(left & right,up+down,10**6),first


@lru_cache(maxsize=96)
def translated_votes(memory,origin,goal,readings,old,prior,sites,focus=None):
    h,w=len(memory),len(memory[0])
    side=max(36,(max(w,h)+1)//2)
    first_counts=[dict.fromkeys(Move,0.) for _ in readings]
    valid_terrain=exact=0
    kinds=(0,1) if focus is None else (focus,)
    if any(not readings[i] or not sites[i] for i in kinds):
        return valid_terrain,exact,first_counts
    if prior is not None and abs(prior[0]-origin[0])+abs(prior[1]-origin[1])!=1:
        return valid_terrain,exact,first_counts
    seen=[(x,y,c!="#") for y,row in enumerate(memory) for x,c in enumerate(row) if c!="?"]
    seen.append((*goal,True))
    if focus is not None:
        def separation(a,b):
            return max(abs(a[0]-b[0]),abs(a[1]-b[1]))
        origins=[(origin,readings[focus])]
        if prior is not None and old[focus] is not None:
            origins.append((prior,old[focus]))
        relevant=tuple(p for p in sites[focus] if any(separation(a,p)<=d for a,d in origins))
        if not relevant:
            return valid_terrain,exact,first_counts
        def nearby(x,y):
            p=x,y
            # Две дополнительные единицы сохраняют клетки, которые могут
            # подтверждать углы диагонали на пути длиной не больше показания.
            return any(separation(a,p)+min(separation(p,q) for q in relevant)<=d+2 for a,d in origins)
        seen=[item for item in seen if nearby(*item[:2]) or separation(origin,item[:2])<=1]
        sites=tuple(relevant if i==focus else () for i in range(2))
    seen.sort(key=lambda p:max(abs(p[0]-origin[0]),abs(p[1]-origin[1])))
    for draw in range(32):
        road,depth,jump,boxes=landscape(side,draw)
        width=road.shape[1]
        ty,tx=np.indices((width-h+1,width-w+1)).reshape(2,-1)
        for x,y,free in seen:
            ok=road[ty+y,tx+x]==free
            if not ok.all():
                ty,tx=ty[ok],tx[ok]
            if not len(tx):
                break
        if not len(tx):
            continue
        origin_ids=(ty+origin[1])*width+tx+origin[0]
        crop=tx,ty,tx+w,ty+h
        def road_to(p):
            target=(ty+p[1])*width+tx+p[0]
            return routes_to(origin_ids,target,depth,jump,boxes,crop)
        matched=(road_to(goal)[0]<10**6) if focus is None else np.ones(len(tx),bool)
        valid_terrain+=int(matched.sum())
        firsts=[]
        for i in kinds:
            markers=sites[i]
            lengths,steps=map(np.array,zip(*(road_to(p) for p in markers)))
            nearest=lengths.min(axis=0)
            matched &= nearest==readings[i]
            if prior is not None and old[i] is not None:
                reverse=prior[0]-origin[0]+width*(prior[1]-origin[1])
                prior_lengths=(lengths+np.where(steps==reverse,-1,1)).min(axis=0)
                matched &= prior_lengths==old[i]
            firsts.append((i,lengths==nearest,steps))
        exact+=int(matched.sum())
        if not matched.any():
            continue
        for i,nearest,steps in firsts:
            counts=first_counts[i]
            shared=nearest[:,matched].sum(axis=0)
            for m in Move:
                dx,dy=m.value
                if dx and dy:
                    continue
                present=nearest[:,matched] & (steps[:,matched]==dx+dy*width)
                counts[m]+=float((present.sum(axis=0)/shared).sum())
    return valid_terrain,exact,first_counts


class TranslatedPaths(StreetTrunk):
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        _,matched,votes=translated_votes(view.memory,view.position,(x+view.dx,y+view.dy),
                                         readings,old,prior,sites)
        return votes if matched else super().direction_votes(view,readings,old,prior,sites)


class LocalCorridors(StreetTrunkReserve):
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        goal=x+view.dx,y+view.dy
        base=super().direction_votes(view,readings,old,prior,sites)
        rows=[]
        for i in range(2):
            _,matched,votes=translated_votes(view.memory,view.position,goal,readings,old,prior,sites,i)
            rows.append(votes[i] if matched else base[i])
        return rows
