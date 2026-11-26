"""Сопоставлять пути к обеим станциям в одном независимом примере дороги."""
from .reference_shells import ConsistentShells
from .standby_turn import RankedStandby


class MissionShells(ConsistentShells):
    onward = True


class StandbyShells(RankedStandby, MissionShells):
    """Для запасного поворота также проверять достижимость второго обслуживания."""
