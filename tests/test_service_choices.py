"""Выбор сервиса проверяется по движениям автомата на уровнях 07–10."""
import unittest
from dataclasses import replace
from math import dist

from maze.automaton import Automaton, Mode
from maze.environment import Environment, Status
from maze.grid import destination, locations, wave
from maze.interface import Move
from maze.levels import catalogue


class ServiceChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = catalogue()

    def drive(self, level):
        env = Environment(level)
        robot = Automaton()
        stops = []
        while env.status == Status.RUNNING:
            view = env.step(robot.choose(env.view))
            x, y = view.position
            tile = level.grid[y][x]
            if tile in "FR":
                self.assertEqual(robot.mode, {"F": Mode.FUEL, "R": Mode.REPAIR}[tile])
                stops.append(tile)
        self.assertEqual(env.status, Status.SUCCESS)
        return "".join(stops)

    def test_only_a_resource_shortage_causes_a_service_visit(self):
        cases = (("07-locks", "F"), ("08-galleries", "R"),
                 ("09-switchbacks", "FR"), ("10-crossroads", "RF"))
        for name, expected in cases:
            with self.subTest(level=name):
                level = self.levels[name]
                reachable = {p for p, _, _ in wave(level.grid, [level.start])}
                for kind in "FR":
                    sites = locations(level.grid, kind)
                    self.assertGreaterEqual(len(sites), 2)
                    for site in sites:
                        self.assertIn(site, reachable)
                        # К каждой станции ведёт единственный разрешённый въезд.
                        exits = [m for m in Move if destination(level.grid, site, m) is not None]
                        self.assertEqual(len(exits), 1)
                self.assertEqual(self.drive(level), expected)
                self.assertEqual(self.drive(replace(level, fuel=100, health=100)), "")
                if len(expected) == 2:
                    self.assertEqual(self.drive(replace(level, fuel=100)), "R")
                    self.assertEqual(self.drive(replace(level, health=100)), "F")

    def test_known_fuel_routes_reverse_coordinate_order(self):
        level = self.levels["07-locks"]
        low = Environment(level)
        full = Environment(replace(level, fuel=100))
        low_robot, full_robot = Automaton(), Automaton()
        for _ in range(24):
            low.step(low_robot.choose(low.view))
            full.step(full_robot.choose(full.view))
        view = low.view
        self.assertEqual(replace(view, fuel=full.fuel), full.view)
        self.assertEqual((view.position, view.fuel), ((6, 6), 10))
        near, far = (4, 3), (11, 5)
        self.assertLess(dist(view.position, near), dist(view.position, far))
        for grid in (view.memory, level.grid):
            paths = {p: d for p, d, _ in wave(grid, [view.position])}
            self.assertEqual((paths[near], paths[far]), (23, 6))
        self.assertEqual(view.fuel_distance, 6)
        self.assertEqual(full_robot.choose(full.view), Move.WEST)
        first = low_robot.choose(view)
        self.assertEqual((low_robot.mode, first), (Mode.FUEL, Move.EAST))
        low.step(first)
        for _ in range(view.fuel_distance - 1):
            low.step(low_robot.choose(low.view))
        self.assertEqual((low.position, low.refuels), (far, 1))
