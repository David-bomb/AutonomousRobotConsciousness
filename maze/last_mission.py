"""После исчерпания безопасной разведки сравнивает цельные поездки с обслуживанием."""
from functools import lru_cache
from heapq import heappop,heappush
from math import inf,isfinite

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .frontier_automaton import _arrive
from .frontier_reserve import _lengths,_optimistic
from .grid import cell,destination,locations
from .interface import Move
from .pattern_roads import PatternRoads,patterns,street_graph
from .pilot_automaton import Mode
from .resource_automaton import service_order
from .road_hypotheses import lattice,networks
from .street_trunk import StreetTrunk


@lru_cache(maxsize=6)
def mission_fields(memory,goal):
    h,w=len(memory),len(memory[0])
    index=lambda p:p[1]*w+p[0]
    sites=[(k,index(p)) for k in "FR" for p in locations(memory,k)]+[("B",index(goal))]
    board,models,weights=patterns(memory,goal)
    rng=np.random.default_rng(42)
    generic=iter(networks(memory,True,24))
    edges=lattice(w,h)
    samples=[]
    for _ in range(24):
        model=models[rng.choice(len(models),p=weights)]
        graph=next(generic) if model is None else street_graph(memory,model,board,edges,rng,index(goal))
        fields=dijkstra(graph,directed=False,unweighted=True,indices=[p for _,p in sites])
        samples.append(fields)
    return sites,samples


def trip_length(origin,stocks,sites,fields,floors,initial_legs=None):
    queue=[(0,origin,*stocks)]
    seen={}
    best=inf
    while queue:
        used,p,f,h=heappop(queue)
        if used>=best:
            break
        previously_seen=seen.setdefault(p,[])
        if any(a>=f and b>=h for a,b in previously_seen):
            continue
        previously_seen.append((f,h))
        for j,((kind,q),field) in enumerate(zip(sites,fields)):
            leg=initial_legs[j] if used==0 and initial_legs is not None else field[p]
            if used==0 and kind in floors:
                leg=max(leg,floors[kind])
            if p==q or leg>min(f,h) or used+leg>=best:
                continue
            leg=int(leg)
            after=_arrive(kind,leg,f,h)
            if after is not None:
                if kind=="B":
                    best=used+leg
                else:
                    heappush(queue,(used+leg,q,*after))
    return best


def mission_move(view,old,default,calibrate=False,survival=False):
    x,y=view.position
    width=len(view.memory[0])
    idx=lambda p:p[1]*width+p[0]
    sites,worlds=mission_fields(view.memory,(x+view.dx,y+view.dy))
    start=idx(view.position)
    prior=None
    if view.moved:
        dx,dy=view.last_move.value
        prior=idx((x-dx,y-dy))
    readings={"F":view.fuel_distance,"R":view.repair_distance}
    floors={k:max(0,d-1) for k,d in readings.items() if d is not None}
    options={}
    for move in Move:
        q=destination(view.memory,view.position,move)
        if q is not None:
            kind=cell(view.memory,*q)
            if kind=="B":
                return move
            after=_arrive(kind,1,view.fuel,view.health)
            if after is not None:
                options[move]=idx(q),after
    if len(options)<2:
        return next(iter(options),default)
    lower=[]
    if calibrate:
        projected=_optimistic(view.memory)
        lower=[_lengths(projected,(q%width,q//width)) for _,q in sites]
    vote=dict.fromkeys(options,0.)
    for fields in worlds:
        mismatch=derivative=0.
        correction={}
        for kind,actual in readings.items():
            ids=[j for j,(symbol,_) in enumerate(sites) if symbol==kind]
            if actual is None or not ids:
                continue
            current=min(fields[j,start] for j in ids)
            mismatch+=abs(current-actual)
            correction[kind]=actual-current
            if isfinite(current) and prior is not None and old[kind] is not None:
                previous=min(fields[j,prior] for j in ids)
                derivative+=abs(previous-current-(old[kind]-actual))
        if not isfinite(mismatch+derivative):
            continue
        lengths={}
        for m,(q,after) in options.items():
            legs=None
            if calibrate:
                point=q%width,q//width
                legs=[max(lower[j].get(point,inf),field[q]+correction.get(kind,0))
                      for j,((kind,_),field) in enumerate(zip(sites,fields))]
            lengths[m]=trip_length(q,after,sites,fields,floors,legs)
        shortest=min(lengths.values())
        if isfinite(shortest):
            weight=1/((1+mismatch)*(1+derivative))**2
            for m,leg in lengths.items():
                if survival:
                    vote[m]+=weight*isfinite(leg)
                else:
                    vote[m]+=weight/(1+leg-shortest)
    if not any(vote.values()):
        return default
    return max(options,key=lambda m:(vote[m],m==default))


class LastMission(PatternRoads):
    def choose(self,view):
        old={k:self._fallback.last_lengths.get(mode) if view.moved else None
             for k,mode in (("F",Mode.FUEL),("R",Mode.REPAIR))}
        move=super().choose(view)
        if self.activity=="Выезд за рубеж возврата":
            revised=mission_move(view,old,move)
            if revised!=move:
                self.activity="Поездка к финишу через доступные сервисы"
                return revised
        return move


class IndividualStops(StreetTrunk):
    """При наличии альтернативной станции сравнивает последующие участки до финиша."""
    def applies(self,view,goal,stops,demand):
        kind="F" if demand.mode==Mode.FUEL else "R"
        projected=_optimistic(view.memory)
        distances=_lengths(projected,view.position)
        affordable=sum(0<distances.get(p,inf)<=min(view.fuel,view.health)
                       for p,k in stops if k==kind)
        return affordable>=2

    def _fallback_move(self,view,goal,stops):
        demand=service_order(view,self._fallback.mode,self.RESERVE)
        old={k:self._fallback.last_lengths.get(mode) if view.moved else None
             for k,mode in (("F",Mode.FUEL),("R",Mode.REPAIR))}
        move=super()._fallback_move(view,goal,stops)
        if demand is None or "несовместимого" in self.activity or not self.applies(view,goal,stops,demand):
            return move
        revised=mission_move(view,old,move,calibrate=True)
        if revised!=move:
            self._fallback.returning=False
            self.activity="Выбор первой остановки с продолжением к финишу"
        return revised


class DeadlineMission(IndividualStops):
    """Сравнивает целые поездки перед решением, для которого не хватает запаса на пробу."""
    def applies(self,view,goal,stops,demand):
        to_goal=_lengths(_optimistic(view.memory),goal).get(view.position,inf)
        return any(d is not None and stock<to_goal and stock-d<2
                   for stock,d in ((view.fuel,view.fuel_distance),(view.health,view.repair_distance)))


class ViabilityVote(StreetTrunk):
    """Сохраняет родительскую команду при равном числе выполнимых продолжений."""
    def _fallback_move(self,view,goal,stops):
        old={k:self._fallback.last_lengths.get(mode) if view.moved else None
             for k,mode in (("F",Mode.FUEL),("R",Mode.REPAIR))}
        demand=service_order(view,self._fallback.mode,self.RESERVE)
        default=super()._fallback_move(view,goal,stops)
        if demand is None or "несовместимого" in self.activity:
            return default
        move=mission_move(view,old,default,calibrate=True,survival=True)
        if move!=default:
            self._fallback.returning=False
            self.activity="Сохранение возможности финиша"
        return move
