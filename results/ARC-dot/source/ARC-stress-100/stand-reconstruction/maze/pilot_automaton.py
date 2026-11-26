"""Исходный автомат D-0008 для воспроизведения архивных траекторий."""
from enum import Enum

from .grid import cell, locations, wave
from .interface import Move, Observation, Point


class Mode(Enum):
    GOAL = "К финишу"
    FUEL = "На заправку"
    REPAIR = "В мастерскую"


def separation(a: Point, b: Point) -> int:
    """Нижняя граница длины пути при восьми направлениях."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class PilotAutomaton:
    def __init__(self, service_offset: int = 0):
        if service_offset < 0:
            raise ValueError("Прибавка к порогу обслуживания должна быть неотрицательной")
        self.service_offset = service_offset
        self.mode = Mode.GOAL

    @property
    def activity(self) -> str:
        """Подпись для наблюдателя."""
        return self.mode.value

    def _need(self, view: Observation, goal_length: int | None) -> Mode:
        if goal_length is not None and goal_length <= min(view.fuel, view.health):
            return Mode.GOAL
        # В исходном опыте запас на исследование подхода равен четырём ходам.
        demands = ((Mode.FUEL, view.fuel, view.fuel_distance),
                   (Mode.REPAIR, view.health, view.repair_distance))
        urgent = []
        for mode, stock, distance in demands:
            if distance is None or distance == 0 or distance > stock:
                continue
            margin = stock - distance
            if margin <= 4 + self.service_offset or mode == self.mode:
                urgent.append((margin, mode))
        return min(urgent, key=lambda item: item[0])[1] if urgent else Mode.GOAL

    def choose(self, view: Observation) -> Move:
        if view.last_move is None:
            self.mode = Mode.GOAL
        grid = view.memory
        x, y = view.position
        goal = (x + view.dx, y + view.dy)
        paths = {p: (d, first) for p, d, first in wave(grid, [view.position])}
        self.mode = self._need(view, paths[goal][0] if goal in paths else None)
        if self.mode == Mode.GOAL:
            targets = [goal]
        else:
            symbol = "F" if self.mode == Mode.FUEL else "R"
            targets = locations(grid, symbol)
        return self._toward(view, paths, targets)

    @staticmethod
    def _toward(view, paths, targets):
        grid = view.memory
        budget = min(view.health, view.fuel)
        reachable = [p for p in targets if p in paths and 0 < paths[p][0] <= budget]
        if reachable:
            best = min(reachable, key=lambda p: paths[p][0])
            return paths[best][1]
        best_score = None
        best_move = None
        for (i, j), (length, first) in paths.items():
            if first is None:
                continue
            if any(cell(grid, i+dx, j+dy) == "?" for dx, dy in (m.value for m in Move)):
                remaining = min(separation((i, j), p) for p in targets)
                score = (length + remaining, remaining, j, i)
                if best_score is None or score < best_score:
                    best_score, best_move = score, first
        if best_move is not None:
            return best_move
        return next((first for _, first in paths.values() if first is not None), Move.NORTH)
