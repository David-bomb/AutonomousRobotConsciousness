"""Одна предполагаемая дорога может предоставить второй станции весь общий префикс."""
from dataclasses import replace
from functools import lru_cache
from random import Random

from .bridge_automaton import suffixes,transitions
from .grid import destination
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions
from .route_witness import agrees,geometry,paths
from .shared_trunk import split_frequency
from .street_trunk import StreetTrunk,alternatives


MOVES=tuple(Move)


def open_step(p,q,move):
    dx,dy=move.value
    return (1<<q)|((1<<(p+dx))|(1<<(q-dx)) if dx and dy else 0)


def draw_branch(view,service,old,path,layers,neighbors,turns,adjacent,split,rng):
    w=len(view.memory[0])
    offsets={m.value:i for i,m in enumerate(MOVES)}
    def heading(p,q):
        return offsets[(q%w-p%w,q//w-p//w)]
    d=service.distance
    legal=possible_directions(view,service,old)
    first=heading(path[0],path[1])
    visited=opened=1<<path[0]
    proposals=[]
    mass=0.
    end=min(d,len(path)-1)
    for t,p in enumerate(path[:end+1]):
        if t and MOVES[first] not in legal:
            break
        last=heading(path[t-1],p) if t else None
        prior=(1-split)**t*(split if t<end else 1.)
        if t==d:
            if layers[d][last,p]:
                mass+=prior
                proposals.append((mass,t,last,p,visited,opened))
            break
        for j,m in enumerate(MOVES):
            q=int(neighbors[j,p])
            if (not t and m not in legal) or (t+1<len(path) and q==path[t+1]):
                continue
            prob=(turns[last,j,p] if t else 1/len(legal))*layers[t+1][j,q]
            if prob<=0 or (visited>>q)&1 or adjacent[q] & (visited ^ (1<<p)):
                continue
            mass+=prior*prob
            proposals.append((mass,t+1,j,q,visited|(1<<q),opened|open_step(p,q,m)))
        if t<end:
            q=path[t+1]
            visited|=1<<q
            opened|=open_step(p,q,MOVES[heading(p,q)])
    if not proposals:
        return None
    threshold=rng.random()*mass
    _,length,last,p,visited,opened=next(row for row in proposals if row[0]>=threshold)
    first=last if length==1 else first
    weight=mass
    for t in range(length,d):
        choices=[]
        total=legal_mass=0.
        for j,m in enumerate(MOVES):
            q=int(neighbors[j,p])
            prob=turns[last,j,p]*layers[t+1][j,q]
            if prob<=0:
                continue
            total+=prob
            if (visited>>q)&1 or adjacent[q] & (visited ^ (1<<p)):
                continue
            legal_mass+=prob
            choices.append((legal_mass,j,q))
        if not choices:
            return None
        threshold=rng.random()*legal_mass
        _,last,q=next(row for row in choices if row[0]>=threshold)
        opened|=open_step(p,q,MOVES[last])
        p=q
        visited|=1<<p
        weight*=legal_mass/total
    return MOVES[first],opened,weight


@lru_cache(maxsize=96)
def paired_votes(view,readings,old,prior,sites):
    stocks=(view.fuel,view.health)
    if any(d is None or not 0<d<=s for d,s in zip(readings,stocks)):
        return None
    h,w=len(view.memory),len(view.memory[0])
    index=lambda p:p[1]*w+p[0]
    origin=index(view.position)
    x,y=view.position
    goal=(x+view.dx,y+view.dy)
    shifts,adjacent=geometry(w,h)
    known=sum(1<<i for i,s in enumerate("".join(view.memory)) if s not in "?#")|1<<index(goal)
    markers=tuple(sum(1<<index(p) for p in group) for group in sites)
    services=[Service(mode,stocks[k],readings[k],sites[k]) for k,mode in enumerate((Mode.FUEL,Mode.REPAIR))]
    rng=Random(42)
    moves=[m for m in MOVES if destination(view.memory,view.position,m)]
    votes=[dict.fromkeys(moves,0.) for _ in readings]
    _,grid=max(alternatives(view.memory,goal),key=lambda row:row[0])
    assumed=replace(view,memory=grid)
    split=split_frequency(view.memory)
    for k,target in enumerate(services):
        other=services[1-k]
        candidates=paths(assumed,target,old[k],origin,adjacent,rng,with_points=True)
        for restricted in (True,False):
            if not candidates:
                break
            neighbors,turns=transitions(grid,restricted)
            layers={t:v.copy() for t,v in suffixes(assumed,other,old[1-k],restricted)}
            found=False
            for move,opened,weight,points in candidates:
                if not agrees(known|opened,shifts,origin,markers,readings,{k}):
                    continue
                trial=draw_branch(assumed,other,old[1-k],points,layers,neighbors,turns,adjacent,split,rng)
                if trial is None:
                    continue
                m,extension,factor=trial
                mask=known|opened|extension
                if not agrees(mask,shifts,origin,markers,readings,{0,1}):
                    continue
                if prior is not None and None not in old and not agrees(mask,shifts,index(prior),markers,old,set()):
                    continue
                found=True
                votes[k][move]+=weight*factor
                votes[1-k][m]+=weight*factor
            if found:
                break
    return votes if any(votes[0].values()) else None


class PairedWitness(StreetTrunk):
    def direction_votes(self,view,readings,old,prior,sites):
        return paired_votes(view,readings,old,prior,sites) or super().direction_votes(view,readings,old,prior,sites)
