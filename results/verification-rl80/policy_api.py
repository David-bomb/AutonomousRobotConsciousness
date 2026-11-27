"""Помощник для адаптера, который выдаёт оценки восьми физических ходов."""
import math

from maze.interface import Move, Observation

ORDER = (Move.NORTH, Move.EAST, Move.SOUTH, Move.WEST, Move.NE, Move.SE, Move.SW, Move.NW)


class GreedyController:
    def __init__(self, scores):
        self.scores = scores

    def choose(self, view: Observation) -> Move:
        values = tuple(float(x) for x in self.scores(view))
        if (len(values) != len(ORDER) or any(math.isnan(v) or v == math.inf for v in values)
                or max(values) == -math.inf):
            raise ValueError("Нужны восемь оценок с хотя бы одним допустимым ходом")
        return ORDER[values.index(max(values))]
