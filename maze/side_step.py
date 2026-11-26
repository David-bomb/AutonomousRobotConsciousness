"""При запасе в один ход проверить два соседних направления без возврата к предыдущей клетке."""
from .grid import destination, locations
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service, known_paths, possible_directions
from .standby_turn import RankedStandby


def lateral_turn(memory, origin, first, second):
    a = destination(memory,origin,first)
    b = destination(memory,origin,second)
    if a is None or b is None or a==b:
        return None
    delta = b[0]-a[0],b[1]-a[1]
    if max(abs(delta[0]),abs(delta[1])) != 1:
        return None
    side = Move(delta)
    return side if destination(memory,a,side)==b else None


class SideStep(RankedStandby):
    change_probe = False
    exactly_two = False

    def __init__(self):
        super().__init__()
        self.side = self.probe = None

    def choose(self,view):
        planned = self.side if view.moved else None
        self.side = self.probe = None
        before = self._fallback.last_lengths.copy()
        move = super().choose(view)
        if planned:
            mode,turn = planned
            i = int(mode==Mode.REPAIR)
            distance = (view.fuel_distance,view.repair_distance)[i]
            if (self._fallback.mode==mode and distance==before.get(mode) and distance
                    and "несовместимого" not in self.activity):
                sites = tuple(locations(view.memory,"FR"[i]))
                known = known_paths(view.memory,view.position)
                if any(known.get(p,(None,))[0]==distance for p in sites):
                    return move
                service = Service(mode,(view.fuel,view.health)[i],distance,sites)
                options = possible_directions(view,service,before[mode])
                if turn in options and move in options:
                    self.activity = "Запасной поворот без возврата — "+mode.value
                    self._fallback.returning = False
                    return turn
        if self.probe:
            mode,first,turn = self.probe
            if self._fallback.mode==mode and move==first:
                self.side = mode,turn
        return move

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        parent = super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        stocks = view.fuel,view.health
        if (readings[kind] is None or min(stocks[kind],stocks[1-kind]-1)-readings[kind]!=1
                or len(options)<2 or self.exactly_two and len(options)!=2):
            return parent
        weights = self.direction_votes(view,readings,old,prior,sites)[kind]
        pairs = []
        for first in options if self.change_probe else (parent,):
            for second in options:
                side = lateral_turn(view.memory,view.position,first,second)
                if side is not None:
                    a,b = weights.get(first,0.),weights.get(second,0.)
                    pairs.append(((a+b,first==parent,a,second==default),first,side))
        if not pairs:
            return parent
        score,first,side = max(pairs,key=lambda row:row[0])
        if first!=parent and score[0]<=weights.get(parent,0.):
            return parent
        self.probe = (Mode.FUEL,Mode.REPAIR)[kind],first,side
        return first


class PairedSideStep(SideStep):
    change_probe = True


class TwoSideStep(SideStep):
    exactly_two = True
