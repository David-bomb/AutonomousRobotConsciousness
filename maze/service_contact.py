"""Разделить дорогу измеренной длины на известный подход и неизвестное продолжение."""
from functools import lru_cache

from .exit_flow import corridor_probabilities
from .frontier_reserve import _lengths, _optimistic
from .grid import destination
from .interface import Move
from .mission_order import MissionOrderReserve


@lru_cache(maxsize=96)
def road_gates(memory, origin):
    known = _lengths(memory, origin)
    projected = _optimistic(memory)
    return tuple(sorted(p for p in known if any(
        (q := destination(projected, p, m)) is not None and q not in known for m in Move)))


@lru_cache(maxsize=96)
def contacts(memory, origin):
    known = _lengths(memory, origin)
    gates = road_gates(memory, origin)
    first = {m: _lengths(memory, q) for m in Move if (q := destination(memory, origin, m)) is not None}
    approaches = {p: tuple(m for m, distances in first.items() if distances[p] + 1 == known[p])
                  for p in gates}
    return gates, approaches, known


def first_contact_votes(memory, origin, sites, distance, previous, prior):
    gates, approaches, known = contacts(memory, origin)
    if not gates or any(known.get(p) == distance for p in sites):
        return {}
    lower = _lengths(_optimistic(memory), origin)
    before = _lengths(memory, prior) if prior is not None and previous is not None else {}
    votes = {}
    for site in sites:
        if site in known or lower.get(site, float("inf")) > distance:
            continue
        # Обратный путь от станции прекращается при первом входе в известный район.
        # Из измеренной длины вычитается реальная длина раскрытого подхода.
        hit = corridor_probabilities(memory, site, gates)
        for p, moves in approaches.items():
            suffix = distance - known[p]
            if not moves or suffix <= 0 or (p in before and before[p] + suffix < previous):
                continue
            mass = max(0., hit[p][suffix] - hit[p][suffix - 1])
            for move in moves:
                votes[move] = votes.get(move, 0.) + mass / len(moves)
    return votes


class ServiceContact(MissionOrderReserve):
    """При запасе менее двух проб выбирает известный подход к наиболее вероятному выходу."""
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        distance = readings[kind]
        if distance is not None and 0 <= (view.fuel, view.health)[kind] - distance < 2:
            votes = first_contact_votes(view.memory, view.position, sites[kind], distance, old[kind], prior)
            if any(votes.get(m, 0.) for m in options):
                return max(options, key=lambda m: (votes.get(m, 0.), m == default))
        return super().rank_directions(view, readings, old, prior, sites, options, kind, default)
