"""Возврат к наведению по реальному показанию во время поездки за границей базы."""
from math import inf

from .frontier_automaton import _arrive, _paths
from .frontier_reserve import _lengths, _optimistic
from .renewable_routes import renewable_routes
from .resource_automaton import service_order
from .street_trunk import StreetTrunk


def can_rejoin(view, stops, roads):
    for site, (kind, need) in renewable_routes(view.memory, stops).items():
        if site not in roads:
            continue
        length = roads[site][0]
        after = _arrive(kind, length, view.fuel, view.health) if length else (view.fuel,view.health)
        if after and after[1 if kind == "F" else 0] >= need:
            return True
    return False


class RemoteStation(StreetTrunk):
    def station_needed(self, view, goal, stops, roads):
        return not can_rejoin(view, stops, roads)

    def _reconsider_sortie(self, view, goal, stops, planned):
        demand = service_order(view, self._fallback.mode, self.RESERVE)
        if demand is None:
            return None
        roads = _paths(view.memory, view.position)
        nearest_known = min((roads.get(q, (inf,))[0] for q in demand.sites), default=inf)
        if demand.distance >= nearest_known or not self.station_needed(view, goal, stops, roads):
            return None
        move = self._fallback_move(view, goal, stops)
        self.activity = "Поиск сервиса после выхода из базы — " + self.activity
        return move


class MandatoryStop(RemoteStation):
    def station_needed(self, view, goal, stops, roads):
        lower = _lengths(_optimistic(view.memory), goal).get(view.position, inf)
        return min(view.fuel, view.health) < lower
