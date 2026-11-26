"""Использовать измеренное сокращение пути как известный первый отрезок подхода к станции."""
from .interface import Move
from .simple_exit_paths import MissionSimplePaths
from .street_trunk import observed_turns


class ApproachHeading(MissionSimplePaths):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        d = readings[kind]
        if not view.moved or d is None or old[kind] != d + 1 or (view.fuel, view.health)[kind] - d >= 2:
            return super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        # От прежней клетки через текущую проходит один из кратчайших путей.
        # К оценке продолжения применяем частоту такого поворота на обследованных дорогах.
        moves = tuple(Move)
        turns = observed_turns(view.memory)[moves.index(view.last_move)]
        raw = self.direction_votes(view, readings, old, prior, sites)[kind]
        return max(options, key=lambda m: (raw.get(m, 0.) * turns[moves.index(m)], m == default))
