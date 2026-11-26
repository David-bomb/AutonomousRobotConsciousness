"""Выбор после неудачной пробы с учётом запасной команды и видимости дороги."""
from .grid import destination
from .sight_automaton import window_unknown
from .standby_turn import RankedStandby
from .unseen_approach import UnseenReturn


class StandbyExploration(RankedStandby, UnseenReturn):
    """Сначала ранее назначенный поворот, в его отсутствие — выход к новому наблюдению."""


class VisibleStandby(StandbyExploration):
    """Когда запасной поворот ведёт в целиком раскрытое окно, пересчитать выбор на месте."""
    def refine_approach(self, view, service, prior_lengths, returning, default):
        if self.ready is not None:
            q = destination(view.memory, view.position, self.ready[1])
            if q is not None and not window_unknown(view.memory, q):
                return UnseenReturn.refine_approach(self, view, service, prior_lengths, returning, default)
        return super().refine_approach(view, service, prior_lengths, returning, default)
