import unittest
from dataclasses import replace
from unittest.mock import patch

from maze.automaton import ResourceAutomaton as Automaton
from maze.pilot_automaton import PilotAutomaton
from maze.resource_automaton import Mode, Service, possible_directions
from maze.environment import Environment, Status
from maze.grid import locations
from maze.interface import Move
from maze.levels import Level
from maze.runner import run
from scripts.audit_automaton import Reachability


class ResourceTests(unittest.TestCase):
    def test_service_offset_shifts_both_resource_boundaries(self):
        # До сервиса 4, Dx до B равен 8. Порог пилота 4, ресурсной версии 16.
        for controller, base in ((PilotAutomaton, 4), (Automaton, 16)):
            for tile, mode in (("F", Mode.FUEL), ("R", Mode.REPAIR)):
                for extra in (2, 3):
                    stock = 4 + base + extra
                    fuel, health = (stock, 100) if tile == "F" else (100, stock)
                    level = Level.parse(f"Граница\n{fuel} {health}\nA...{tile}...B")
                    view = Environment(level).view
                    for offset in (0, 2, 3):
                        with self.subTest(controller=controller, tile=tile, extra=extra, offset=offset):
                            robot = controller(service_offset=offset)
                            robot.choose(view)
                            self.assertEqual(robot.mode, mode if offset >= extra else Mode.GOAL)

    def test_switch_to_fuel_keeps_the_previous_actual_fuel_reading(self):
        env = Environment(Level.parse("После ремонта\n16 6\nF......A..R.......B"))
        bot = Automaton()
        for _ in range(3):
            env.step(bot.choose(env.view))
        self.assertEqual((env.repairs, env.view.fuel_distance), (1, 10))
        # Предыдущая клетка имела длину 9 до F. Обратный ход доказан даже
        # при неизвестной дороге к заправке, поэтому разведка не требуется.
        with patch("maze.resource_automaton.possible_directions", side_effect=AssertionError("лишняя разведка")):
            move = bot.choose(env.view)
        self.assertEqual((bot.mode, move), (Mode.FUEL, Move.WEST))
        self.assertEqual(env.step(move).fuel_distance, 9)

    def test_first_service_must_leave_other_resource_alive(self):
        level = Level.parse("Сначала ремонт\n8 3\nB..F..A.R")
        env = Environment(level)
        bot = Automaton()
        bot.mode = Mode.FUEL
        self.assertEqual(bot.choose(env.view), Move.EAST)
        self.assertEqual(bot.mode, Mode.REPAIR)
        self.assertEqual(run(env, bot), Status.SUCCESS)

    def test_confirmed_route_requires_repair_then_fuel(self):
        level = Level.parse("Порядок обслуживания\n6 4\nF.A.R....B")
        env = Environment(level)
        bot = Automaton()
        stops = []
        while env.status == Status.RUNNING:
            # Сценарий с полностью изученной дорогой, как после разведки.
            move = bot.choose(replace(env.view, memory=level.grid))
            env.step(move)
            symbol = level.grid[env.position[1]][env.position[0]]
            if symbol in "FR":
                stops.append(symbol)
        self.assertEqual(env.status, Status.SUCCESS)
        self.assertEqual(stops[:2], ["R", "F"])

    def test_actual_reading_eliminates_an_impossible_first_step(self):
        level = Level.parse("Дорога вдоль стены\n10 90\nA...F.B\n.......")
        env = Environment(level)
        before = env.view.fuel_distance
        view = env.step(Move.EAST)
        station = Service(Mode.FUEL, view.fuel, view.fuel_distance, tuple(locations(view.memory, "F")))
        options = possible_directions(view, station, before)
        self.assertIn(Move.EAST, options)
        self.assertNotIn(Move.WEST, options)
        self.assertNotIn(Move.SOUTH, options)

    def test_reachability_respects_last_unit_and_episode_limit(self):
        level = Level.parse("Ресурсы\n2 2\nAF.B")
        oracle = Reachability(level)
        self.assertTrue(oracle.possible(level.start, 1, 3, 3))
        self.assertFalse(oracle.possible(level.start, 1, 2, 3))
        self.assertFalse(oracle.possible(level.start, 1, 3, 2))
