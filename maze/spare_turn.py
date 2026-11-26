"""Оставить возможность сменить направление при одном запасном ходе до сервиса."""
from .grid import destination
from .interface import Move
from .service_exit_paths import StationUsefulPaths


def protected_directions(memory, origin, options):
    cells = {m: destination(memory, origin, m) for m in options}
    result = {}
    for first, point in cells.items():
        adjacent = {destination(memory, point, m) for m in Move}
        result[first] = {m for m, q in cells.items() if q == point or q in adjacent}
    return result


class ProtectedTurn(StationUsefulPaths):
    strict = True

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        d = readings[kind]
        own, other = (view.fuel, view.health) if kind == 0 else (view.health, view.fuel)
        if d is None or min(own - d, other - d - 1) != 1:
            return parent
        covered = protected_directions(view.memory, view.position, options)
        candidates = [m for m in options if not self.strict or covered[m] > covered[parent]]
        if not candidates:
            return parent
        evidence = self.direction_votes(view, readings, old, prior, sites)[kind]
        # Если r начинает кратчайший путь длины d и из q до r один разрешённый ход,
        # через q и r до станции можно доехать за 1+1+(d-1) шагов.
        def rank(m):
            return sum(evidence.get(n, 0.) for n in options if n in covered[m]), m == parent, evidence.get(m, 0.)
        best = max(candidates, key=rank)
        return best if rank(best) > rank(parent) else parent


class SpareTurn(ProtectedTurn):
    strict = False
