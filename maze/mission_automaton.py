"""Голосуют только подходы к сервису, после которых в гипотезе остаётся финиш."""
from functools import lru_cache
from math import ceil,isfinite

from scipy.sparse.csgraph import dijkstra

from .frontier_automaton import _arrive
from .grid import destination
from .interface import Move
from .road_hypotheses import RangeRoads,networks


def completion(first,f,h,sites,fields,goal,other_floor):
    queue=[(first,f,h,True)]
    seen={}
    while queue:
        s,f,h,initial=queue.pop()
        if fields[s,goal]<=min(f,h):
            return True
        stocks=seen.setdefault(s,[])
        if any(a>=f and b>=h for a,b in stocks):
            continue
        stocks.append((f,h))
        for j,(kind,idx) in enumerate(sites):
            if s==j:
                continue
            length=fields[s,idx]
            if initial and kind!=sites[first][0]:
                length=max(length,other_floor)
            after=_arrive("FR"[kind],length,f,h)
            if after:
                queue.append((j,*after,False))
    return False


@lru_cache(maxsize=96)
def mission_votes(view,readings,old,prior,by_kind):
    width=len(view.memory[0])
    index=lambda p:p[1]*width+p[0]
    center=index(view.position)
    goal=index((view.position[0]+view.dx,view.position[1]+view.dy))
    neighbors={m:index(q) for m in Move if (q:=destination(view.memory,view.position,m))}
    sites=[(i,index(p)) for i,group in enumerate(by_kind) for p in group]
    ids=[p for _,p in sites]
    generic=[dict.fromkeys(neighbors,0.) for _ in range(2)]
    viable=[dict.fromkeys(neighbors,0.) for _ in range(2)]
    for graph in networks(view.memory):
        fields=dijkstra(graph,directed=False,unweighted=True,indices=ids)
        near=[min((field[center] for field,(kind,_) in zip(fields,sites) if kind==i),default=float("inf"))
              for i in range(2)]
        if any(d is not None and not isfinite(near[i]) for i,d in enumerate(readings)):
            continue
        err=sum(abs(near[i]-d) for i,d in enumerate(readings) if d is not None)
        if prior is not None:
            for i,prev in enumerate(old):
                if prev is not None and readings[i] is not None:
                    before=min(field[index(prior)] for field,(kind,_) in zip(fields,sites) if kind==i)
                    err+=abs(before-near[i]-(prev-readings[i]))
        weight=1/(1+err)**2
        for kind,actual in enumerate(readings):
            if not actual or actual>min(view.fuel,view.health):
                continue
            all_first=set()
            feasible=set()
            for s,(i,_) in enumerate(sites):
                if i!=kind or fields[s,center]!=near[i]:
                    continue
                candidates={m for m,q in neighbors.items() if fields[s,q]==near[i]-1}
                all_first.update(candidates)
                after=_arrive("FR"[i],actual,view.fuel,view.health)
                onward=max(0,(readings[1-i] or 0)-actual)
                if after and completion(s,*after,sites,fields,goal,onward):
                    feasible.update(candidates)
                    continue
                # Другой сервис может встретиться непосредственно на одной
                # из кратчайших дорог к первоначальной остановке.
                for t,(j,idx) in enumerate(sites):
                    if i==j or not isfinite(fields[s,idx]) or not near[i]:
                        continue
                    tail=ceil(fields[s,idx]*actual/near[i])
                    if not 0<tail<actual or (readings[j] or 0)>actual-tail:
                        continue
                    stocks=(100,100-tail) if i==0 else (100-tail,100)
                    if min(stocks)<=0 or not completion(s,*stocks,sites,fields,goal,onward):
                        continue
                    feasible.update(m for m in candidates if fields[t,neighbors[m]]+fields[s,idx]==near[i]-1)
            for m in all_first:
                generic[kind][m]+=weight/len(all_first)
            for m in feasible:
                viable[kind][m]+=weight/len(feasible)
    return [v if any(v.values()) else g for v,g in zip(viable,generic)]


class MissionRoads(RangeRoads):
    direction_votes=staticmethod(mission_votes)
