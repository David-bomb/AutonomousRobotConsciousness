"""Отдельная оценка направления, когда на ошибочный выход с возвратом не хватает запаса."""
from .frontier_automaton import _paths
from .grid import locations
from .pattern_roads import pattern_votes
from .pilot_automaton import Mode
from .proven_approach import certified
from .resource_automaton import opposite,possible_directions,service_order
from .street_trunk import StreetTrunkReserve


class NoProbeStreets(StreetTrunkReserve):
    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        if (view.fuel,view.health)[kind]-readings[kind] >= 2:
            return super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        x,y=view.position
        score=pattern_votes(view.memory,view.position,readings,old,prior,sites,(x+view.dx,y+view.dy),256)[kind]
        return max(options,key=lambda m:(score[m],m==default))


class LastDirection(NoProbeStreets):
    def _fallback_move(self,view,goal,stops):
        old=tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        service=service_order(view,self._fallback.mode,self.RESERVE)
        default=super()._fallback_move(view,goal,stops)
        if service is None or service.stock-service.distance>=2 or "несовместимого" in self.activity:
            return default
        kind=int(service.mode==Mode.REPAIR)
        if old[kind] is not None and old[kind]<service.distance:
            return opposite(view.last_move)
        roads=_paths(view.memory,view.position)
        if any(roads.get(site,(None,))[0]==service.distance for site in service.sites):
            return default
        sites=tuple(tuple(locations(view.memory,k)) for k in "FR")
        prior=None
        if view.moved:
            dx,dy=view.last_move.value
            prior=view.position[0]-dx,view.position[1]-dy
        certain,possible=certified(view.memory,view.position,service.distance,old[kind],prior,sites[kind])
        choices=possible_directions(view,service,old[kind])
        options={m for m in choices if m in (certain or possible)} or choices
        if not options:
            return default
        # Порядок перебора физических команд стабилен и при равных оценках.
        ordered=[m for m in choices if m in options]
        next_move=self.rank_directions(view,(view.fuel_distance,view.repair_distance),old,prior,
                                       sites,ordered,kind,default)
        if next_move!=default:
            self.activity="Наведение без запаса на пробу — "+service.mode.value
            self._fallback.returning=False
        return next_move
