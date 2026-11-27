"""Искусственная политика для проверки канала. Журнал вызовов читает только тест."""
from importlib import import_module
import json
import sys

from maze.interface import Move


def load(checkpoint: bytes):
    wrapper = import_module("results.verification-rl80.previous_readings.controller").PreviousReadings

    def decide(view, old_fuel, old_repair):
        fuel, repair = view.fuel_distance, view.repair_distance
        print(json.dumps([old_fuel, old_repair, fuel, repair]), file=sys.stderr)
        if checkpoint == b"pair":
            if old_fuel is None and old_repair is None:
                return Move.EAST
            # Начальные дороги могут изменить каждое прежнее показание независимо.
            return {(False, False): Move.EAST, (True, False): Move.NORTH,
                    (False, True): Move.SOUTH, (True, True): Move.WEST}[
                        old_fuel > fuel, old_repair > repair]
        if checkpoint == b"cycle":
            return Move.WEST if view.position == (3, 1) else Move.EAST
        if checkpoint == b"collision":
            return Move.NORTH if fuel == 0 and view.moved else Move.EAST
        raise ValueError("Неизвестный тест")

    return wrapper(decide)
