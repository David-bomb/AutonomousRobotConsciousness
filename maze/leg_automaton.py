"""Выбор очередности двух обслуживаний по реальным замерам из текущей клетки."""
from .frontier_automaton import _paths
from .frontier_reserve import ReserveFrontier
from .grid import locations
from .interface import Move
from .pilot_automaton import Mode, separation
from .resource_automaton import Service, opposite, possible_directions, service_order


def order_from_origin(view, active):
    """Верхняя граница пути к двум ближайшим сервисам через текущую клетку."""
    choices = []
    for mode, symbol, stock, other, distance, onward in (
        (Mode.FUEL, "F", view.fuel, view.health, view.fuel_distance, view.repair_distance),
        (Mode.REPAIR, "R", view.health, view.fuel, view.repair_distance, view.fuel_distance),
    ):
        if distance is None or onward is None or not 0 < distance <= stock or distance >= other:
            continue
        if 100 - stock <= distance:
            continue
        # Переход от первого сервиса ко второму через точку принятия решения.
        # На первом сервисе пополняется только один из двух ресурсов.
        leg = distance + onward
        slack = min(stock - distance, other - distance - leg, 99 - leg)
        sites = tuple(p for p in locations(view.memory, symbol)
                      if separation(view.position, p) <= distance)
        rank = (slack >= 0, min(slack, 0), mode == active, -distance)
        choices.append((rank, Service(mode, stock, distance, sites)))
    return max(choices, key=lambda item: item[0])[1] if choices else None


def toward_service(view, service, old, returning):
    roads = _paths(view.memory, view.position)
    for site in service.sites:
        if roads.get(site, (None,))[0] == service.distance:
            return roads[site][1], False
    if old is not None and view.moved and service.distance > old:
        return opposite(view.last_move), True
    options = possible_directions(view, service, old)
    if not options:
        return None, False
    if returning and view.last_move:
        moves = list(Move)
        index = moves.index(opposite(view.last_move))
        return next(m for m in moves[index+1:] + moves[:index+1] if m in options), False
    return min(options, key=options.get), False


class AnchorOrder(ReserveFrontier):
    """При выборе между сервисами сначала проверяет выполнимость их пары."""
    def _fallback_move(self, view, goal, stops):
        active = self._fallback.mode
        previous = self._fallback.last_lengths.copy()
        default_service = service_order(view, active, self.RESERVE)
        move = super()._fallback_move(view, goal, stops)
        if default_service is None:
            return move
        service = order_from_origin(view, active)
        if service is None or service.mode == default_service.mode:
            return move
        direction, returning = toward_service(view, service, previous.get(service.mode), False)
        if direction is None:
            return move
        self._fallback.mode = service.mode
        self._fallback.returning = returning
        self.activity = "Пара сервисов через текущую клетку — " + service.mode.value
        return direction
