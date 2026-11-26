import unittest
from dataclasses import replace
from unittest.mock import patch

from maze.environment import Environment
from maze.frontier_reserve import ReserveFrontier
from maze.interface import Move
from maze.levels import Level
from maze.resource_automaton import Mode, possible_directions, service_order
from maze.sight_automaton import Probe, SightDetour, SightRoute, with_wall


def small_maze():
    return Environment(Level.parse("""Окно к заправке
10 32
#########
#...##..#
#.A.R...#
#...##..#
##.#...##
#B......#
#...F#..#
#########"""))


class SightTests(unittest.TestCase):
    def test_window_separates_shortest_approaches(self):
        view = small_maze().view
        service = service_order(view, Mode.GOAL, 2)
        base = ReserveFrontier().choose(view)
        robot = SightRoute()
        probe = robot._select_probe(view, service, None, base)
        self.assertEqual((base, robot.choose(view)), (Move.SW, Move.SOUTH))
        self.assertEqual(probe.distinguishing_cells, ((2, 4), (3, 4)))
        original = possible_directions(view, service, None).keys()
        for point in probe.distinguishing_cells:
            changed = replace(view, memory=with_wall(view.memory, point))
            remaining = possible_directions(changed, service, None).keys()
            self.assertTrue(remaining and remaining < original)

    def test_budget_must_cover_return(self):
        view = small_maze().view
        tight = replace(view, fuel=view.fuel_distance + 1)
        self.assertEqual(SightRoute().choose(tight), ReserveFrontier().choose(tight))

    def test_off_route_window_can_distinguish_a_passage(self):
        env = Environment(Level.parse("""Боковой обзор
20 44
#########
#...#B..#
#....R.##
#.....#F#
#A...#..#
#..#....#
#.###...#
#########"""))
        for move in (Move.NE, Move.SE, Move.NE, Move.NE):
            env.step(move)
        view = env.view
        service = service_order(view, Mode.FUEL, 2)
        args = view, service, None, Move.EAST
        self.assertIsNone(SightRoute()._select_probe(*args))
        self.assertEqual(SightDetour()._select_probe(*args), Probe(Move.SOUTH, True, ((6, 4),)))

    def test_executed_side_step_is_reversed_once(self):
        env = small_maze()
        robot = SightDetour()
        with patch.object(robot, "_select_probe", return_value=Probe(Move.WEST, True, ())):
            self.assertEqual(robot.choose(env.view), Move.WEST)
        env.step(Move.WEST)
        self.assertEqual(robot.choose(env.view), Move.EAST)
        self.assertFalse(robot._return_from_probe)
        env.step(Move.EAST)
        self.assertEqual(env.position, env.level.start)
