"""Запасная команда может зависеть от второго датчика во время физической пробы."""
from .interface import Move
from .pilot_automaton import Mode
from .probe_pairs import paired_directions, two_attempts
from .standby_exploration import StandbyExploration
from .standby_turn import remaining_turns


class ConditionalStandby(StandbyExploration):
    select_probe = False

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        probe = super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        if self.offer is None or len(options)<3 or readings[1-kind] is None:
            return probe
        mode,_,reserve = self.offer
        pairs = paired_directions(view,readings,old,prior,sites)
        matrix = pairs.T if kind else pairs
        moves = tuple(Move)
        legal = [moves.index(m) for m in options]
        if self.select_probe:
            values = {m:two_attempts(matrix,legal,moves.index(m))[0] for m in options}
            candidate = max(options,key=lambda m:(values[m],m==probe))
            alternatives = remaining_turns(view.memory,view.position,options,candidate)
            if candidate != probe and alternatives:
                probe = candidate
                reserve = max(alternatives,key=lambda m:(matrix[moves.index(m)].sum(),m==reserve))
        remaining = remaining_turns(view.memory,view.position,options,probe)
        by_outcome = []
        col = moves.index(probe)
        for shorter in (False,True):
            weights = {}
            for m in remaining:
                row = matrix[moves.index(m)]
                weights[m] = row[col] if shorter else row.sum()-row[col]
            best = max(remaining,key=lambda m:(weights[m],m==reserve)) if any(v>0 for v in weights.values()) else reserve
            by_outcome.append(best)
        # Храним только команды для трёх исходов будущего физического замера.
        self.offer = mode,probe,(reserve,*by_outcome)
        return probe

    def saved_turn(self, view, mode, reserve, before):
        if isinstance(reserve,Move):
            return reserve
        other = Mode.REPAIR if mode == Mode.FUEL else Mode.FUEL
        now = view.repair_distance if other == Mode.REPAIR else view.fuel_distance
        previous = before.get(other)
        if previous is None or now is None or now == previous:
            return reserve[0]
        return reserve[2 if now < previous else 1]


class ConditionalProbe(ConditionalStandby):
    """Выбрать также саму пробу с учётом двух возможных продолжений."""
    select_probe = True
