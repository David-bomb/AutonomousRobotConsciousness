"""Разовый выбор подтверждённого заезда при уходе из возобновляемого района."""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite

from .conditional_mission import required_after_stop
from .deadline_goal import goal_deadline
from .frontier_automaton import _arrive,_paths
from .grid import cell,destination,locations,wave
from .interface import Move, Point
from .last_mission import mission_fields
from .outer_automaton import OuterService
from .pilot_automaton import Mode
from .remote_station import can_rejoin
from .resource_automaton import confirmed_route
from .sight_automaton import window_unknown
from .street_trunk import StreetTrunkReserve


@lru_cache(maxsize=8)
def continuations(memory,goal):
    sites,fields=mission_fields(memory,goal)
    return sites,[(model,required_after_stop(sites,model)) for model in fields]


@dataclass(frozen=True)
class Stop:
    kind: str
    point: Point
    length: int
    move: Move
    stocks: tuple[int, int]


def nearest_known(view,sites,goal):
    roads=_paths(view.memory,view.position)
    best={}
    for p,k in sites:
        d,move=roads.get(p,(0,None))
        if d and (after:=_arrive(k,d,view.fuel,view.health)) and (
                k not in best or d<best[k].length):
            best[k]=Stop(k,p,d,move,after)
    return list(best.values())


@lru_cache(maxsize=96)
def first_legs(memory,origin,goal):
    markers = frozenset(p for kind in "FR" for p in locations(memory,kind)) | {goal}
    return [(p,d,move) for p,d,move in wave(memory,[origin],stop_at=markers)
            if d and p != goal and p in markers]


def known_first_stops(view,sites,goal):
    allowed = dict(sites)
    return [Stop(allowed[p],p,d,move,stock)
            for p,d,move in first_legs(view.memory,view.position,goal)
            if p in allowed and (stock := _arrive(allowed[p],d,view.fuel,view.health))]


def supported_stop(view,goal,choices):
    if not choices:
        return None
    sites,models=continuations(view.memory,goal)
    width=len(view.memory[0])
    index=lambda p:p[1]*width+p[0]
    origin=index(view.position)
    ids={k:[j for j,(kind,_) in enumerate(sites) if kind==k] for k in "FR"}
    arrival={index(stop.point):stop for stop in choices}
    scores=dict.fromkeys(choices,0.)
    direct=0.
    for fields,need in models:
        error=sum(abs(min(fields[j,origin] for j in ids[k])-d) for k,d in
                  zip("FR",(view.fuel_distance,view.repair_distance)) if ids[k] and d is not None)
        if not isfinite(error):
            continue
        weight=1/(1+error)**2
        if fields[-1,origin]<=min(view.fuel,view.health):
            direct+=weight
        for j,(kind,site) in enumerate(sites[:-1]):
            stop=arrival.get(site)
            if stop and stop.stocks[1 if kind=="F" else 0]>=need[j]:
                scores[stop]+=weight
    best=max(choices,key=lambda stop:(scores[stop],-stop.length))
    return best if scores[best]>direct else None


class KnownLanding(StreetTrunkReserve):
    before_crossing=False
    landing=None
    choices = staticmethod(nearest_known)

    def ongoing_stop(self,view,goal,sites):
        return next(iter(self.choices(view,sites,goal)),None)

    def _drive(self,view,kind,move):
        self.landing=kind
        self._fallback.mode=Mode.FUEL if kind=="F" else Mode.REPAIR
        self._fallback.last_lengths={Mode.FUEL:view.fuel_distance,Mode.REPAIR:view.repair_distance}
        self._fallback.returning=False
        self.activity="Подтверждённый заезд на последнем выезде — "+self._fallback.mode.value
        return move

    def choose(self,view):
        if view.last_move is None or cell(view.memory,*view.position) in "FR":
            self.landing=None
        if self.landing:
            x,y=view.position
            goal=x+view.dx,y+view.dy
            roads=_paths(view.memory,view.position)
            if confirmed_route(view,goal,roads) is None:
                sites=[(p,self.landing) for p in locations(view.memory,self.landing)]
                stop=self.ongoing_stop(view,goal,sites)
                if stop:
                    return self._drive(view,stop.kind,stop.move)
            self.landing=None
        return super().choose(view)

    def _reconsider_sortie(self,view,goal,stops,planned):
        if goal_deadline(view):
            return None
        p=destination(view.memory,view.position,planned) if self.before_crossing else None
        if p and p==goal:
            return None
        after=_arrive(cell(view.memory,*p),1,view.fuel,view.health) if p else None
        check=replace(view,position=p,fuel=after[0],health=after[1]) if after else view
        # can_rejoin использует только известные дороги, позицию и два запаса.
        if can_rejoin(check,stops,_paths(view.memory,check.position)):
            return None
        stop=supported_stop(view,goal,self.choices(view,stops,goal))
        return self._drive(view,stop.kind,stop.move) if stop else None


class BoundaryLanding(KnownLanding):
    before_crossing=True


class FirstLanding(BoundaryLanding):
    """Сравнивает все станции, достижимые первой остановкой по раскрытой дороге."""
    choices = staticmethod(known_first_stops)

    def ongoing_stop(self,view,goal,sites):
        return supported_stop(view,goal,self.choices(view,sites,goal))


class AfterLanding(FirstLanding):
    """После целевого обслуживания завершает хотя бы один выезд к новой клетке."""
    exploring_after_stop = False
    after_stop_rank = None

    @property
    def outer_rank(self):
        return self.after_stop_rank if self.exploring_after_stop else None

    def choose(self,view):
        if view.last_move is None:
            self.exploring_after_stop=False
        elif self.landing==cell(view.memory,*view.position):
            self.exploring_after_stop=True
        move=super().choose(view)
        if self.exploring_after_stop:
            q=destination(view.memory,view.position,move)
            if q is not None and window_unknown(view.memory,q):
                self.exploring_after_stop=False
        return move

    def _reconsider_sortie(self,view,goal,stops,planned):
        if self.exploring_after_stop:
            return None
        return super()._reconsider_sortie(view,goal,stops,planned)


class OutwardAfterLanding(AfterLanding):
    """После обслуживания оценивает продолжение снаружи открытой области."""
    after_stop_rank = staticmethod(OuterService.outer_rank)
