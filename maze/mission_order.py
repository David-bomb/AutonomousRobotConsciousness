"""Выбор вида первой станции по возможности продолжить поездку после заезда."""
from functools import lru_cache
from math import inf, isfinite

from .frontier_automaton import _arrive
from .frontier_reserve import _lengths, _optimistic
from .last_mission import mission_fields, trip_length
from .leg_automaton import toward_service
from .pilot_automaton import Mode
from .resource_automaton import Service, service_order
from .room_roads import RoomPassages, room_predicts_memory
from .street_trunk import StreetTrunk


def first_stop_models(view, goal, old):
    """Возвращает вес геометрии и возможные ближайшие станции обоих видов."""
    roads = _lengths(_optimistic(view.memory), view.position)
    width = len(view.memory[0])
    x, y = view.position
    origin = y*width+x
    prior = None
    if view.moved:
        dx, dy = view.last_move.value
        prior = (y-dy)*width+x-dx
    actual = view.fuel_distance, view.repair_distance
    sites, worlds = mission_fields(view.memory, goal)
    groups = [[j for j,(kind, _) in enumerate(sites) if kind == k] for k in "FR"]
    if None in actual or not all(groups):
        return
    for fields in worlds:
        model = [min(fields[j, origin] for j in group) for group in groups]
        if not all(isfinite(v) for v in model):
            continue
        error = sum(abs(d - m) for d, m in zip(actual, model))
        delta = sum(abs(min(fields[j,prior] for j in group) - m - (previous - d))
                    for group, m, previous, d in zip(groups, model, old, actual)
                    if prior is not None and previous is not None)
        weight = 1 / ((1+error)*(1+delta))**2
        nearest=[]
        for k,group in enumerate(groups):
            points=[]
            for j in group:
                q=sites[j][1]
                if fields[j,origin]==model[k] and roads.get((q%width,q//width),inf)<=actual[k]:
                    points.append(q)
            nearest.append(points)
        yield weight,fields,sites,nearest


@lru_cache(maxsize=128)
def order_support(view, goal, old):
    arrivals=[_arrive(kind,d,view.fuel,view.health) if d is not None else None
              for kind,d in zip("FR",(view.fuel_distance,view.repair_distance))]
    votes=[0.,0.]
    for weight,fields,sites,nearest in first_stop_models(view,goal,old):
        for k,points in enumerate(nearest):
            if arrivals[k] and any(isfinite(trip_length(q,arrivals[k],sites,fields,{})) for q in points):
                votes[k]+=weight
    return votes


class MissionOrder(StreetTrunk):
    def _fallback_move(self, view, goal, stops):
        old = self._fallback.last_lengths.copy()
        parent_service = service_order(view, self._fallback.mode, self.RESERVE)
        default = super()._fallback_move(view, goal, stops)
        lengths = view.fuel_distance, view.repair_distance
        if parent_service is None or "несовместимого" in self.activity or any(
                d is None or d <= 0 or _arrive(k,d,view.fuel,view.health) is None
                for k,d in zip("FR", lengths)):
            return default
        modes = (Mode.FUEL, Mode.REPAIR)
        previous = tuple(old.get(m) if view.moved else None for m in modes)
        votes = order_support(view, goal, previous)
        parent = modes.index(parent_service.mode)
        k = 1-parent
        if votes[k] <= votes[parent]:
            return default
        stock = (view.fuel, view.health)[k]
        if 100-stock <= lengths[k]:
            return default
        roads = _lengths(_optimistic(view.memory), view.position)
        sites = tuple(p for p,s in stops if s == "FR"[k] and roads.get(p,inf) <= lengths[k])
        service = Service(modes[k], stock, lengths[k], sites)
        direction, returning = toward_service(view, service, previous[k], False)
        if direction is None:
            return default
        self._fallback.mode = modes[k]
        self._fallback.returning = returning
        self.activity = "Первый заезд с продолжением до финиша — " + modes[k].value
        return self.refine_approach(view, service, old, returning, direction)


class MissionOrderReserve(MissionOrder):
    """После выбора вида сервиса учитывает запас на переход между двумя станциями."""
    budget = True


class RoomMission(MissionOrderReserve, RoomPassages):
    """Выбирает вид сервиса по продолжению поездки и учитывает проходы между помещениями."""


class SurveyedRoomMission(RoomMission):
    """Применяет гипотезу о помещениях, когда она лучше предсказывает уже обследованную область."""
    def direction_votes(self, view, readings, old, prior, sites):
        if room_predicts_memory(view.memory):
            return super().direction_votes(view, readings, old, prior, sites)
        return StreetTrunk.direction_votes(self, view, readings, old, prior, sites)
