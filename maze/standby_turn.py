"""Назначить одну запасную команду до пробы, пока доступен замер предыдущего подхода."""
from .grid import destination
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import known_paths, opposite, possible_directions
from .second_window import ReturnWindows


def remaining_turns(memory, origin, options, probe):
    q = destination(memory, origin, probe)
    # Если длина после пробы вырастет с d до d+1, в одном ходе от q
    # не может быть клетки, из которой до сервиса осталось d-1.
    adjacent = {destination(memory, q, m) for m in Move}
    return {m:v for m,v in options.items() if m != probe and destination(memory, origin, m) not in adjacent}


class StandbyTurn(ReturnWindows):
    allow_multiple = False

    def saved_turn(self, view, mode, reserve, before):
        return reserve

    def __init__(self):
        super().__init__()
        self.standby = self.ready = self.offer = None

    def choose(self, view):
        attempted = self.standby if view.moved else None
        before = self._fallback.last_lengths
        if view.last_move is None:
            self.ready = None
        self.standby = self.offer = None
        move = super().choose(view)
        self.ready = None
        if attempted:
            mode, reserve = attempted
            now = view.fuel_distance if mode == Mode.FUEL else view.repair_distance
            old = before.get(mode)
            if (self._fallback.mode == mode and now is not None and old is not None and now > old
                    and move == opposite(view.last_move)):
                self.ready = mode, self.saved_turn(view,mode,reserve,before)
        if self.offer:
            mode, probe, reserve = self.offer
            if self._fallback.mode == mode and move == probe:
                self.standby = mode, reserve
        return move

    def refine_approach(self, view, service, prior_lengths, returning, default):
        parent = super().refine_approach(view, service, prior_lengths, returning, default)
        if self.ready is None or service is None or not returning or not view.moved or self.ready[0] != service.mode:
            return parent
        old = prior_lengths.get(service.mode)
        options = possible_directions(view, service, old)
        reserve = self.ready[1]
        roads = known_paths(view.memory, view.position)
        if (reserve in options and parent in options and "несовместимого" not in self.activity
                and not any(roads.get(p,(None,))[0] == service.distance for p in service.sites)):
            self.activity = "Запасной ход после неудачной пробы — " + service.mode.value
            return reserve
        return parent

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        own, other = (view.fuel,view.health) if kind == 0 else (view.health,view.fuel)
        if prior is None or readings[kind] is None or min(own,other-1) - readings[kind] not in (2,3):
            return parent
        remaining = remaining_turns(view.memory, view.position, options, parent)
        if len(remaining) == 1 or remaining and self.allow_multiple:
            reserve = next(iter(remaining))
            if len(remaining) > 1:
                reserve = super().rank_directions(view, readings, old, prior, sites, remaining, kind,
                                                  default if default in remaining else reserve)
            self.offer = (Mode.FUEL, Mode.REPAIR)[kind], parent, reserve
        return parent


class RankedStandby(StandbyTurn):
    allow_multiple = True
