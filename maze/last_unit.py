"""Отбрасывает ход, когда он доказуемо исчерпывает ресурс до сервиса и финиша."""
from .frontier_reserve import _lengths,_optimistic
from .grid import destination,locations
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions
from .road_hypotheses import RangeRoads


def surviving_moves(view,old,default):
    goal = view.position[0]+view.dx,view.position[1]+view.dy
    available = {m:q for m in Move if (q:=destination(view.memory,view.position,m))}
    permitted = set(available)
    advice = {}
    prior = None
    if view.moved:
        dx,dy = view.last_move.value
        prior = view.position[0]-dx,view.position[1]-dy
    for i,(mode,kind,stock,d) in enumerate(((Mode.FUEL,"F",view.fuel,view.fuel_distance),
                                            (Mode.REPAIR,"R",view.health,view.repair_distance))):
        if d is None or stock-d>1:
            continue
        to_goal = _lengths(_optimistic(view.memory),goal)
        service = Service(mode,stock,d,tuple(locations(view.memory,kind)))
        options = possible_directions(view,service,old[i]) if d==stock else {}
        for m,q in available.items():
            if q==goal or 1+to_goal.get(q,float("inf"))<=stock:
                continue
            if (d>stock or (d==stock and m not in options) or
                    (q==prior and old[i] is not None and old[i]>stock-1)):
                permitted.discard(m)
            elif m in options:
                advice[m]=advice.get(m,0)+1
    if default in permitted or not permitted:
        return default
    return max((m for m in Move if m in permitted),key=lambda m:advice.get(m,0))


class LastUnit(RangeRoads):
    def choose(self,view):
        old = tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        default=super().choose(view)
        move=surviving_moves(view,old,default)
        if move!=default:
            self.activity="Последняя единица запаса до сервиса"
            self._fallback.returning=False
        return move
