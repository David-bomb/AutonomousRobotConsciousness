import unittest
from dataclasses import replace

from maze.automaton import Automaton, Mode
from maze.environment import Environment, Status
from maze.grid import destination, locations, wave
from maze.interface import Move
from maze.levels import Level, catalogue
from maze.runner import advance, run


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = catalogue()

    def test_progression(self):
        for name, level in self.levels.items():
            with self.subTest(name=name):
                env = Environment(level)
                self.assertEqual(run(env, Automaton()), Status.SUCCESS)
                self.assertEqual(env.position, level.goal)
                if name.startswith(("04", "06")):
                    self.assertGreaterEqual(env.refuels, 1)
                if name.startswith(("05", "06")):
                    self.assertGreaterEqual(env.repairs, 1)

    def test_lengths_use_roads_and_do_not_reveal_them(self):
        env = Environment(self.levels["04-fuel"])
        view = env.view
        paths = {p: d for p, d, _ in wave(env.level.grid, [view.position])}
        self.assertEqual([paths[p] for p in locations(view.memory, "F")], [26, 6])
        self.assertEqual(view.fuel_distance, 6)
        self.assertIsNone(view.repair_distance)
        self.assertEqual(view.memory[1][3], "?")
        env.step(Move.EAST)
        self.assertEqual(env.view.fuel_distance, 5)
        self.assertEqual(env.view.memory[1][3], ".")
        self.assertEqual(view.memory[1][3], "?")
        self.assertEqual(env.reset(), view)

    def test_unreachable_service_and_independent_minima(self):
        env = Environment(Level.parse("test\n100 100\nA.R#F\n..B##"))
        self.assertIsNone(env.view.fuel_distance)
        self.assertEqual(env.view.repair_distance, 2)
        self.assertEqual(env.view.window, ("###", "#..", "#.."))
        env.step(Move.EAST)
        env.step(Move.EAST)
        self.assertEqual(env.view.repair_distance, 0)
        self.assertEqual(env.health, 100)

    def test_identical_input_except_remembered_road(self):
        def after_trip(out, back):
            env = Environment(self.levels["02-loop"])
            for move in ([Move.NORTH] + [out]*3 + [Move.NORTH]*2 + [back]*2 +
                         [out]*2 + [Move.SOUTH]*2 + [back]*3 + [Move.SOUTH, Move.NORTH]):
                self.assertTrue(env.step(move).moved)
            return env.view

        left = after_trip(Move.WEST, Move.EAST)
        right = after_trip(Move.EAST, Move.WEST)
        self.assertEqual(replace(left, memory=right.memory), right)
        self.assertNotEqual(left.memory, right.memory)
        self.assertEqual(Automaton().choose(left), Move.WEST)
        self.assertEqual(Automaton().choose(right), Move.EAST)

    def test_eight_moves_and_blocked_corner(self):
        grid = ("...", "...", "...")
        self.assertEqual(len({destination(grid, (1, 1), m) for m in Move}), 8)
        env = Environment(Level.parse("test\n10 10\nA#B\n..."))
        view = env.step(Move.SE)
        self.assertFalse(view.moved)
        self.assertEqual(view.last_move, Move.SE)
        self.assertEqual((view.position, view.fuel, view.health), ((0, 0), 9, 9))

    def test_last_unit_service_finish_and_failures(self):
        for row, resources, status in (("AFB", "1 5", Status.RUNNING),
                                       ("AB", "1 1", Status.SUCCESS),
                                       ("A.B", "1 5", Status.EMPTY),
                                       ("A.B", "5 1", Status.BROKEN)):
            with self.subTest(row=row, resources=resources):
                env = Environment(Level.parse(f"test\n{resources}\n{row}"))
                env.step(Move.EAST)
                self.assertEqual(env.status, status)
                if status == Status.RUNNING:
                    self.assertEqual((env.fuel, env.view.fuel_distance), (100, 0))

    def test_only_observation_is_needed_by_replacement(self):
        class East:
            def choose(self, view):
                return Move.EAST

        env = Environment(Level.parse("test\n100 100\nA..B"), limit=2)
        self.assertEqual(run(env, East()), Status.LIMIT)
        with self.assertRaises(RuntimeError):
            env.step(Move.EAST)

    def test_both_service_modes_change_the_route(self):
        env = Environment(self.levels["06-expedition"])
        robot = Automaton()
        modes = set()
        while env.status == Status.RUNNING:
            advance(env, robot)
            modes.add(robot.mode)
        self.assertEqual(modes, set(Mode))
        self.assertEqual((env.refuels, env.repairs), (1, 1))

    def test_crossroads_explores_dead_ends_before_finding_finish(self):
        level = self.levels["10-crossroads"]
        env = Environment(level)
        robot = Automaton()
        finish_search = []
        while env.status == Status.RUNNING:
            advance(env, robot)
            if robot.mode == Mode.GOAL:
                finish_search.append(env.position)
            else:
                finish_search.clear()
        self.assertEqual(env.status, Status.SUCCESS)

        entrance = (27, 3)
        first = finish_search.index(entrance)
        last = len(finish_search) - 1 - finish_search[::-1].index(entrance)
        excursion = finish_search[first:last + 1]
        for tip in ((32, 3), (31, 5), (29, 6)):
            with self.subTest(tip=tip):
                exits = [destination(level.grid, tip, move) for move in Move]
                index = excursion.index(tip)
                self.assertEqual([p for p in exits if p is not None], [excursion[index - 1]])
                self.assertEqual(excursion[index + 1], excursion[index - 1])
        self.assertEqual(finish_search[last + 1], (27, 2))
