"""Совместный начальный отрезок двух дорог измеренной длины."""
from functools import lru_cache

import numpy as np

from .bridge_automaton import suffixes,transitions
from .grid import destination
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions
from .road_hypotheses import RangeRoads,field_votes


@lru_cache(maxsize=96)
def split_frequency(grid):
    forks=observed=0
    for y,row in enumerate(grid[1:-1],1):
        for x,c in enumerate(row[1:-1],1):
            if c in "#?" or any("?" in grid[j][x-1:x+2] for j in range(y-1,y+2)):
                continue
            observed+=1
            forks+=sum(destination(grid,(x,y),m) is not None for m in Move)>2
    return (forks+1)/(observed+2)


@lru_cache(maxsize=96)
def trunk_votes(view,readings,old,prior,sites,budget,require_match=False,turn_weights=None,split_prior=None,
                independent_exits=False, first_kind=None, joint_first=False, split_profile=None, cross_links=()):
    grid=view.memory
    stocks=view.fuel,view.health
    def fallback():
        return None if require_match else field_votes(grid,view.position,readings,old,prior,sites)
    if any(d is None or not 0<d<=s for d,s in zip(readings,stocks)):
        return fallback()
    services=[Service(mode,stocks[i],readings[i],sites[i]) for i,mode in enumerate((Mode.FUEL,Mode.REPAIR))]
    allowed=[possible_directions(view,s,p) for s,p in zip(services,old)]
    start=view.position[1]*len(grid[0])+view.position[0]
    ranks=np.arange(len(Move))[:,None]
    moves=tuple(Move)
    prior_split=split_frequency(grid) if split_prior is None else split_prior
    for restricted in (True,False):
        neighbors,T=transitions(grid,restricted,turn_weights)
        layers=[{t:v.copy() for t,v in suffixes(view,s,p,restricted,turn_weights)} for s,p in zip(services,old)]
        f,r=layers
        end=min(readings)
        # Если один сервис лежит прямо на дороге к другому, дальнейшая
        # часть принадлежит только второй дороге.
        joint=f[end]*r[end]
        distinct=np.maximum(0.,1-np.sum(T*T,axis=1))
        def branch_at(t):
            prob = split_profile[t] if split_profile is not None and t < len(split_profile) else prior_split
            branch = np.where(distinct>1e-10,prob,0.)
            return branch*distinct if independent_exits else branch
        branch = branch_at(0)
        norm=np.where(distinct>1e-10,distinct,1.)
        def can_split(t):
            between=sum(readings)-2*t
            if budget and first_kind is not None:
                return between < 100 and readings[first_kind] + between <= stocks[1-first_kind]
            if not budget or any(readings[i]+between<=stocks[1-i] for i in range(2)):
                return 1.
            return cross_links[t] if t < len(cross_links) else 0.
        for t in range(end-1,0,-1):
            if split_profile is not None:
                branch = branch_at(t)
            left=f[t+1][ranks,neighbors]
            right=r[t+1][ranks,neighbors]
            common=np.einsum("ijn,jn->in",T,joint[ranks,neighbors])
            split=np.zeros_like(common)
            support = can_split(t)
            if support:
                same=np.einsum("ijn,ijn,jn->in",T,T,left*right)
                split=(np.maximum(0.,f[t][:,:-1]*r[t][:,:-1]-same)/norm)*support
            joint[:,:-1]=((1-branch)*common+branch*split)*((f[t][:,:-1]>0)&(r[t][:,:-1]>0))
        legal=[i for i,m in enumerate(moves) if destination(grid,view.position,m) is not None]
        first=[[ls[1][i,neighbors[i,start]] if m in allow else 0. for i,m in enumerate(moves)]
               for ls,allow in zip(layers,allowed)]
        vote=[dict.fromkeys((moves[i] for i in legal),0.) for _ in range(2)]
        if len(legal)<2:
            return fallback()
        origin_prior=split_profile[0] if split_profile else prior_split
        origin_split=origin_prior*(1-1/len(legal)) if independent_exits else origin_prior
        if joint_first:
            pairs = np.zeros((len(moves), len(moves)))
            for i in legal:
                m = moves[i]
                if m in allowed[0] and m in allowed[1]:
                    pairs[i, i] = (1-origin_split)*joint[i,neighbors[i,start]]
                if can_split(0):
                    for j in legal:
                        if i != j:
                            pairs[i, j] = origin_split*first[0][i]*first[1][j]/(len(legal)-1)*can_split(0)
            if pairs.any():
                return pairs
            continue
        for i in legal:
            m=moves[i]
            for k in range(2):
                if m in allowed[k] and m in allowed[1-k]:
                    vote[k][m]+=(1-origin_split)*joint[i,neighbors[i,start]]
                if can_split(0):
                    vote[k][m]+=origin_split*first[k][i]*sum(
                        first[1-k][j] for j in legal if j!=i)/(len(legal)-1)*can_split(0)
        if all(any(out.values()) for out in vote):
            return vote
    return fallback()


class SharedTrunk(RangeRoads):
    budget=False

    def direction_votes(self,view,readings,old,prior,sites):
        return trunk_votes(view,readings,old,prior,sites,self.budget)


class TrunkReserve(SharedTrunk):
    budget=True
