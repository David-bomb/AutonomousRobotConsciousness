"""Оценка разведывательного выезда с пополнениями вдоль предполагаемой дороги к цели."""
from functools import lru_cache

from .deadline_goal import goal_deadline
from .exit_flow import eventual_exits
from .frontier_automaton import _arrive
from .frontier_reserve import _lengths
from .second_window import ReturnWindows
from .sensed_paths import road_samples


def trip_fits(length, stops, stocks, known_approach, readings):
    last = 0
    for t, kind in stops:
        measured = readings[int(kind == "R")]
        if measured is not None and known_approach + t < measured:
            return False
        stocks = _arrive(kind,t-last,*stocks)
        if stocks is None:
            return False
        last = t
    return length-last <= min(stocks)


@lru_cache(maxsize=256)
def extra_share(memory, goal, points, position, readings, point, fuel, health):
    shares = eventual_exits(memory,goal,points)
    if not shares[point][-1]:
        return 0.
    sample = road_samples(memory,goal,points)
    approach = _lengths(memory,position)[point]
    count = successes = 0
    for p,length,_,stops in sample:
        if p == point:
            count += 1
            if stops and length > min(fuel,health):
                successes += trip_fits(length,stops,(fuel,health),approach,readings)
    return shares[point][-1]*successes/count if count else 0.


class OnwardServices(ReturnWindows):
    def chance(self, view, goal, point, fuel, health, curves):
        direct = super().chance(view,goal,point,fuel,health,curves)
        if goal_deadline(view):
            return direct
        return direct + extra_share(view.memory,goal,tuple(curves),view.position,
                                    (view.fuel_distance,view.repair_distance),point,fuel,health)
