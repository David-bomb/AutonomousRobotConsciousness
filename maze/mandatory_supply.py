"""Сохранить подтверждённый путь к ресурсу, без которого финиш уже недостижим."""
from math import inf

from .frontier_automaton import _arrive
from .frontier_reserve import _lengths, _optimistic, _relaxed_route
from .grid import cell, destination, locations
from .pilot_automaton import Mode
from .resource_automaton import known_paths
from .resource_automaton import Service, possible_directions
from .route_shield import completion_status
from .standby_turn import RankedStandby


class MandatoryProof(RankedStandby):
    require_proof = True

    def choose(self,view):
        previous = self._fallback.last_lengths.copy()
        chosen = super().choose(view)
        x,y = view.position
        goal = x+view.dx,y+view.dy
        gauges = view.fuel_distance,view.repair_distance
        stocks = view.fuel,view.health
        urgent = [k for k,d in enumerate(gauges) if d is not None and d>0 and 0<=stocks[k]-d<2]
        if not urgent:
            return chosen
        optimistic = _optimistic(view.memory)
        lower = _lengths(optimistic,goal).get(view.position,inf)
        urgent = [k for k in urgent if lower>stocks[k]]
        if not urgent:
            return chosen
        q = destination(view.memory,view.position,chosen)
        if q is None:
            return chosen
        stops = tuple((p,k) for k in "FR" for p in locations(view.memory,k)) + ((goal,"B"),)
        known = known_paths(view.memory,view.position)
        next_stocks = _arrive(cell(view.memory,*q),1,*stocks)
        old = tuple(previous.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        best = None
        for index in urgent:
            symbol = "FR"[index]
            for p,k in stops:
                if k!=symbol or p not in known:
                    continue
                length,move = known[p]
                after = _arrive(k,length,*stocks)
                if not after:
                    continue
                # После подтверждённого заезда должен хотя бы оставаться
                # возможен финиш при свободных неизвестных клетках.
                floors = {s:max(0,d-length) for s,d in zip("FR",gauges) if d is not None}
                if not _relaxed_route(optimistic,p,goal,*after,stops,floors):
                    continue
                alternative = _lengths(view.memory,p).get(q,inf)
                if q==p or (next_stocks and alternative<inf and _arrive(k,alternative,*next_stocks)):
                    return chosen
                key = stocks[index]-gauges[index],length,p
                if best is None or key<best[0]:
                    best = key,move,index
        if best is None:
            return chosen
        if self.require_proof:
            prior = None
            if view.moved:
                dx,dy = view.last_move.value
                prior = x-dx,y-dy
            status,_,_ = completion_status(view,goal,gauges,prior,old,chosen)
            if status!="impossible":
                return chosen
        _,move,index = best
        self.standby = self.ready = self.offer = None
        self._fallback.mode = (Mode.FUEL,Mode.REPAIR)[index]
        self._fallback.returning = False
        self._fallback.last_lengths = dict(zip((Mode.FUEL,Mode.REPAIR),gauges))
        self.activity = "Подтверждённый путь к обязательной станции"
        return move


class ConfirmedSupply(MandatoryProof):
    require_proof = False


class ResourceDeadline(MandatoryProof):
    """При нулевом запасе все ходы должны сокращать длину к обязательному сервису."""
    prefer_descent = False

    def choose(self,view):
        previous = self._fallback.last_lengths.copy()
        move = super().choose(view)
        gauges = view.fuel_distance,view.repair_distance
        stocks = view.fuel,view.health
        tight = [i for i,d in enumerate(gauges) if d is not None and stocks[i]==d and d>0]
        if not tight:
            return move
        x,y = view.position
        optimistic = _lengths(_optimistic(view.memory),view.position)
        to_goal = optimistic.get((x+view.dx,y+view.dy),inf)
        tight = [i for i in tight if to_goal>stocks[i]]
        if not tight:
            return move
        modes = Mode.FUEL,Mode.REPAIR
        old = tuple(previous.get(mode) if view.moved else None for mode in modes)
        sites = tuple(tuple(locations(view.memory,k)) for k in "FR")
        required = None
        for i in tight:
            marks = tuple(p for p in sites[i] if optimistic.get(p,inf)<=gauges[i])
            service = Service(modes[i],stocks[i],gauges[i],marks)
            candidates = possible_directions(view,service,old[i])
            required = dict(candidates) if required is None else {m:d for m,d in required.items() if m in candidates}
        if not required:
            return move
        if move in required and (not self.prefer_descent or self._fallback.mode in {modes[i] for i in tight}):
            return move
        index = tight[0]
        prior = None
        if view.moved:
            dx,dy = view.last_move.value
            prior = x-dx,y-dy
        default = move if move in required else next(iter(required))
        new = self.rank_directions(view,gauges,old,prior,sites,required,index,default) if len(required)>1 else default
        if new==move:
            return move
        self.ready = self.standby = self.offer = None
        self._fallback.mode = modes[index]
        self._fallback.returning = False
        self.activity = "Безусловное сокращение пути к обязательной станции"
        return new


class DirectedDeadline(ResourceDeadline):
    """При выборе другой потребности сравнивать все допустимые обязательные спуски."""
    prefer_descent = True
