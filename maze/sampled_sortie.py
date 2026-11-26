"""Выбрать доступный выезд, из которого в предполагаемой геометрии удаётся доехать до цели."""
from functools import lru_cache
from math import isfinite

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .conditional_mission import possible_worlds,required_after_stop
from .excursion_automaton import lookouts
from .frontier_automaton import _arrive,_paths,_service_states
from .grid import cell
from .pilot_automaton import Mode
from .road_hypotheses import lattice
from .street_trunk import StreetTrunkReserve


@lru_cache(maxsize=6)
def passages(memory,goal,root):
    h,w=len(memory),len(memory[0])
    index=lambda p:p[1]*w+p[0]
    component=_paths(memory,root)
    if goal in component:
        return (),()
    outside=np.ones(w*h,dtype=bool)
    outside[[index(p) for p in component]]=False
    frontiers=sorted(set(lookouts(memory,goal)).intersection(component))
    gateway={index(p):p for p in frontiers}
    sites,worlds=possible_worlds(memory,goal)
    targets=[s for s in sites if outside[s[1]]]
    groups=[[j for j,(kind,_) in enumerate(sites) if kind==k] for k in "FR"]
    edges=lattice(w,h)
    results=[]
    for opened,fields,_ in worlds:
        u,v=edges[:2,opened[edges].all(axis=0)]
        interior=outside[u] & outside[v]
        graph=coo_matrix((np.ones(interior.sum()),(u[interior],v[interior])),shape=(w*h,w*h)).tocsr()
        lengths=dijkstra(graph,directed=False,unweighted=True,indices=[q for _,q in targets])
        needs=required_after_stop(targets,lengths)
        links={}
        for a,b in ((u,v),(v,u)):
            crossing=~outside[a]&outside[b]
            for start,end in zip(a[crossing],b[crossing]):
                if start in gateway:
                    links.setdefault(gateway[start],[]).append(end)
        onward={p:1+lengths[:,ids].min(axis=1) for p,ids in links.items()}
        gauges=[fields[ids].min(axis=0) if ids else None for ids in groups]
        results.append((gauges,onward,needs))
    return targets,results


class SampledSortie(StreetTrunkReserve):
    def _reconsider_sortie(self,view,goal,stops,planned):
        memory=view.memory
        targets,worlds=passages(memory,goal,min(_paths(memory,view.position)))
        if not worlds:
            return None
        origin=view.position[1]*len(memory[0])+view.position[0]
        weights=[]
        for gauges,_,_ in worlds:
            error=sum(abs(field[origin]-d) for field,d in zip(gauges,(view.fuel_distance,view.repair_distance))
                      if field is not None and d is not None)
            weights.append(1/(1+error)**2 if isfinite(error) else 0.)
        @lru_cache(None)
        def evidence(p,f,h):
            success=0.
            for weight,(_,onward,need) in zip(weights,worlds):
                legs=onward.get(p)
                if not weight or legs is None:
                    continue
                if legs[-1]<=min(f,h):
                    success+=weight
                    continue
                for j,(kind,_) in enumerate(targets[:-1]):
                    length=legs[j]
                    if length<=min(f,h):
                        after=_arrive(kind,int(length),f,h)
                        if after and after[1 if kind=="F" else 0]>=need[j]:
                            success+=weight
                            break
            return success
        candidates=lookouts(memory,goal)
        best=None
        for traveled,f,h,first,roads in _service_states(memory,view.position,view.fuel,view.health,stops):
            for p,lower in candidates.items():
                if p==view.position or p not in roads:
                    continue
                d,move=roads[p]
                after=_arrive(cell(memory,*p),d,f,h) if d else (f,h)
                if after is None:
                    continue
                weight=evidence(p,*after)
                if weight:
                    rank=(weight,min(after)-lower,-lower,-traveled-d,(first or move)==planned)
                    if best is None or rank>best[0]:
                        best=rank,first or move
        if best:
            self._fallback.mode=Mode.GOAL
            self._fallback.last_lengths={Mode.FUEL:view.fuel_distance,Mode.REPAIR:view.repair_distance}
            self._fallback.returning=False
            self.activity="Выезд с возможным обслуживанием за границей базы"
            return best[1]
        return None
