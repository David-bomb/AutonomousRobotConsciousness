"""Физика и единый канал восприятия. Контроллер получает только Observation."""
from enum import Enum

from .grid import cell, destination, locations, wave
from .interface import Grid, Move, Observation
from .levels import Level
from .memory import CellMemory


class Status(Enum):
    RUNNING = "В пути"
    SUCCESS = "Финиш"
    EMPTY = "Топливо исчерпано"
    BROKEN = "Исправность исчерпана"
    LIMIT = "Лимит ходов"


class Environment:
    def __init__(self, level: Level, limit: int | None = None):
        self.level = level
        self.limit = limit if limit is not None else 8 * len(level.grid) * len(level.grid[0])
        if self.limit < 1:
            raise ValueError("Лимит должен быть положительным")
        self._lengths = {}
        self._services = {}
        for symbol in "FR":
            sites = locations(level.grid, symbol)
            self._services.update(dict.fromkeys(sites, symbol))
            self._lengths[symbol] = {p: d for p, d, _ in wave(level.grid, sites)}
        self.reset()

    def reset(self) -> Observation:
        self.position = self.level.start
        self.fuel, self.health = self.level.fuel, self.level.health
        self.steps = self.refuels = self.repairs = 0
        self.last_move = self.moved = None
        self.status = Status.RUNNING
        self._memory = CellMemory(len(self.level.grid[0]), len(self.level.grid), self._services)
        return self._observe()

    def _observe(self) -> Observation:
        x, y = self.position
        window: Grid = tuple("".join(cell(self.level.grid, i, j) for i in range(x - 1, x + 2))
                             for j in range(y - 1, y + 2))
        memory = self._memory.update(self.position, window)
        self.view = Observation(memory, window, self.position, self.level.goal[0] - x,
                                self.level.goal[1] - y, self._lengths["F"].get(self.position),
                                self._lengths["R"].get(self.position), self.fuel, self.health,
                                self.last_move, self.moved)
        return self.view

    def step(self, move: Move) -> Observation:
        if self.status != Status.RUNNING:
            raise RuntimeError("Эпизод завершён")
        target = destination(self.level.grid, self.position, move)
        self.last_move, self.moved = move, target is not None
        self.steps += 1
        self.fuel -= 1
        self.health -= 1
        if target is not None:
            self.position = target
            service = cell(self.level.grid, *target)
            if service == "F":
                self.fuel = 100
                self.refuels += 1
            elif service == "R":
                self.health = 100
                self.repairs += 1
        if self.position == self.level.goal:
            self.status = Status.SUCCESS
        elif self.fuel == 0:
            self.status = Status.EMPTY
        elif self.health == 0:
            self.status = Status.BROKEN
        elif self.steps >= self.limit:
            self.status = Status.LIMIT
        return self._observe()
