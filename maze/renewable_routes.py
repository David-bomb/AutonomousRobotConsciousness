"""Какой второй запас нужен после заезда, чтобы доехать до возобновляемой пары."""
from functools import lru_cache

from .excursion_automaton import shelters
from .frontier_automaton import _paths
from .street_trunk import StreetTrunk


@lru_cache(maxsize=128)
def renewable_routes(memory, stops):
    known = dict(shelters(memory, stops))
    legs = {p: _paths(memory, p) for p, _ in stops}
    # После заезда запас своего типа полон. Достаточно хранить минимальное
    # требование ко второму запасу. Улучшения целочисленны и ограничены.
    while True:
        changed = False
        for p, kind in stops:
            best = known.get(p, (kind, 101))[1]
            for q, (other, need) in tuple(known.items()):
                if q == p or q not in legs[p]:
                    continue
                length = legs[p][q][0]
                if kind == other:
                    required = length + need
                elif 100 - length >= need:
                    required = length
                else:
                    continue
                if required < best:
                    best = required
                    known[p] = kind, required
                    changed = True
        if not changed:
            return known


class RenewableRoutes(StreetTrunk):
    return_sites = staticmethod(renewable_routes)
