"""Применить две точные длины к нескольким возможным рисункам неизвестных улиц."""
from dataclasses import replace
from functools import lru_cache

import numpy as np

from .pattern_roads import MarkedStreets,ManyStreets,pattern_votes,patterns
from .grid import destination
from .frontier_reserve import _lengths,_optimistic
from .interface import Move
from .shared_trunk import split_frequency, trunk_votes


@lru_cache(maxsize=128)
def alternatives(memory,goal):
    board,models,weights=patterns(memory,goal)
    h,w=len(memory),len(memory[0])
    y,x=np.indices((h,w)).reshape(2,-1)
    out=[]
    for j in np.argsort(-weights)[:4]:
        model=models[j]
        if model is None:
            out.append((weights[j],memory))
            continue
        sx,sy,ox,oy=model
        hidden=(board==b"?") & (x%sx!=ox) & (y%sy!=oy)
        hidden[goal[1]*w+goal[0]]=False
        hypothesized=board.copy()
        hypothesized[hidden]=b"#"
        out.append((weights[j],tuple(row.tobytes().decode() for row in hypothesized.reshape(h,w))))
    return tuple(out)


class StreetTrunk(MarkedStreets):
    adapt_turns = False
    independent_exits = False
    road_patterns = staticmethod(alternatives)
    budget = False
    split_profile = None
    cross_links = None

    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        goal=(x+view.dx,y+view.dy)
        scores=[{},{}]
        turn_weights=observed_turns(view.memory) if self.adapt_turns else None
        split_prior=split_frequency(view.memory) if self.adapt_turns else None
        profile = self.split_profile(view, readings, old, prior, sites) if self.split_profile is not None else None
        links = self.cross_links(view, readings, old, prior, sites) if self.cross_links is not None else ()
        for weight,grid in self.road_patterns(view.memory,goal):
            opinion=trunk_votes(replace(view,memory=grid),readings,old,prior,sites,self.budget,True,
                                turn_weights,split_prior,self.independent_exits,split_profile=profile,cross_links=links)
            if opinion is None:
                continue
            for i in range(2):
                for move,p in opinion[i].items():
                    scores[i][move]=scores[i].get(move,0.)+weight*p
        return scores if all(any(row.values()) for row in scores) else super().direction_votes(
            view,readings,old,prior,sites)


@lru_cache(maxsize=128)
def observed_turns(memory):
    moves=tuple(Move)
    ring=((0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1))
    def angle(a,b):
        delta=abs(ring.index(a.value)-ring.index(b.value))
        return min(delta,8-delta)
    # Частота на одно направление. У прямого хода один вариант,
    # у поворотов на 45, 90 и 135 градусов по два.
    multiplicity=(1,2,2,2,1)
    counts=[2,2,2,2,1]
    for y,row in enumerate(memory[1:-1],1):
        for x,symbol in enumerate(row[1:-1],1):
            if symbol in "#?" or any("?" in line[x-1:x+2] for line in memory[y-1:y+2]):
                continue
            p=(x,y)
            neighbors={m:q for m in moves if (q:=destination(memory,p,m)) is not None}
            for back,source in neighbors.items():
                heading=Move((-back.value[0],-back.value[1]))
                options=[]
                for out,target in neighbors.items():
                    if target==source:
                        continue
                    offset=(target[0]-source[0],target[1]-source[1])
                    direct=next((m for m in moves if m.value==offset),None)
                    if direct is None or destination(memory,source,direct) is None:
                        options.append(angle(heading,out))
                for turn in options:
                    counts[turn]+=1/len(options)
    return tuple(tuple(counts[angle(a,b)]/multiplicity[angle(a,b)] for b in moves) for a in moves)


class AdaptiveTrunk(StreetTrunk):
    """Оценивает частоту поворотов и разветвлений по раскрытой памяти эпизода."""
    adapt_turns=True


class IndependentExits(StreetTrunk):
    """Даже при разветвлении две дороги могут продолжиться через один выход."""
    independent_exits=True


class StreetTrunkReserve(StreetTrunk):
    """Проверяет запас на переход между станциями через предполагаемое место расхождения."""
    budget=True


@lru_cache(maxsize=128)
def contains_cycle(memory,origin):
    visited={origin}
    queue=[(origin,None)]
    while queue:
        p,parent=queue.pop()
        for move in Move:
            q=destination(memory,p,move)
            if q is None or q==parent:
                continue
            if q in visited:
                return True
            visited.add(q)
            queue.append((q,p))
    return False


class CycleGate(StreetTrunk):
    """Проверяет модель разветвления на дереве, пока наблюдения допускают дерево."""
    def direction_votes(self,view,readings,old,prior,sites):
        if contains_cycle(view.memory,view.position):
            x,y=view.position
            return pattern_votes(view.memory,view.position,readings,old,prior,sites,
                                 (x+view.dx,y+view.dy),ManyStreets.samples)
        return super().direction_votes(view,readings,old,prior,sites)


class MarginGate(StreetTrunk):
    """Обращается к путям измеренной длины, когда неверная проба уже не оплачивается."""
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        target_distance=_lengths(_optimistic(view.memory),(x+view.dx,y+view.dy)).get(view.position,float("inf"))
        exposed=any(d is not None and stock<target_distance and stock-d<2
                    for d,stock in zip(readings,(view.fuel,view.health)))
        if exposed:
            return super().direction_votes(view,readings,old,prior,sites)
        return MarkedStreets.direction_votes(self,view,readings,old,prior,sites)
