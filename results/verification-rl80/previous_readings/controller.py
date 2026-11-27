"""Память двух показаний для отдельного варианта контроллера E-0016."""
from maze.interface import Move, Observation


class PreviousReadings:
    __slots__ = ("_decide", "_fuel", "_repair")

    def __init__(self, decide):
        self._decide = decide
        self.reset()

    def reset(self):
        self._fuel = self._repair = None

    def choose(self, view: Observation) -> Move:
        move = self._decide(view, self._fuel, self._repair)
        if type(move) is not Move:
            raise TypeError("Политика должна вернуть один физический ход")
        self._fuel, self._repair = view.fuel_distance, view.repair_distance
        return move
