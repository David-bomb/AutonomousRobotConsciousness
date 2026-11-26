"""Закончить выбранный заезд, пока оба ресурса допускают его достижение."""
from .excursion_automaton import shelters
from .frontier_automaton import _paths
from .grid import cell,locations
from .leg_automaton import toward_service
from .pilot_automaton import Mode,separation
from .resource_automaton import Service,confirmed_route,possible_directions,service_order
from .road_hypotheses import RangeRoads


class Homing(RangeRoads):
    def __init__(self):
        super().__init__()
        self.target=None
        self.probe_return=False

    def choose(self,view):
        if view.last_move is None:
            self.target=None
            self.probe_return=False
            self._fallback=self._make_fallback()
        roads=_paths(view.memory,view.position)
        goal=view.position[0]+view.dx,view.position[1]+view.dy
        stops=tuple((p,k) for k in "FR" for p in locations(view.memory,k))
        if (goal in roads and confirmed_route(view,goal,roads)) or roads.keys() & shelters(view.memory,stops).keys():
            self.target=None
            self.probe_return=False
            return super().choose(view)
        active=self.target
        if active is not None:
            symbol="F" if active==Mode.FUEL else "R"
            if cell(view.memory,*view.position)==symbol:
                self.target=None
        if self.target is None:
            order=service_order(view,Mode.GOAL,self.RESERVE)
            self.target=order.mode if order else None
        if self.target is None:
            return super().choose(view)
        i=int(self.target==Mode.REPAIR)
        stocks=(view.fuel,view.health)
        readings=(view.fuel_distance,view.repair_distance)
        d=readings[i]
        if d is None or d>stocks[i] or d>=stocks[1-i] or d==0:
            self.target=None
            self.probe_return=False
            return super().choose(view)
        all_sites=tuple(tuple(p for p,k in stops if k==symbol) for symbol in "FR")
        service=Service(self.target,stocks[i],d,tuple(p for p in all_sites[i] if separation(p,view.position)<=d))
        old=tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        default,returning=toward_service(view,service,old[i],self.probe_return and active==self.target)
        if default is None:
            return super().choose(view)
        self.probe_return=returning
        self._fallback.last_lengths=dict(zip((Mode.FUEL,Mode.REPAIR),readings))
        self._fallback.mode=self.target
        self._fallback.returning=returning
        self.activity="Достижение выбранного сервиса — "+self.target.value
        if returning or any(roads.get(p,(None,))[0]==d for p in service.sites):
            return default
        options=possible_directions(view,service,old[i])
        if len(options)<2 or default not in options:
            return default
        prior=None
        if view.moved:
            dx,dy=view.last_move.value
            prior=view.position[0]-dx,view.position[1]-dy
        scores=self.direction_votes(view,readings,old,prior,all_sites)[i]
        return max(options,key=lambda m:(scores[m],m==default))
