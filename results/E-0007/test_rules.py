"""Малые сценарии для границ ресурсов, возврата после пробы и выбора нового окна."""
from dataclasses import replace
import unittest

from run import HERE, setup
setup(HERE.parents[1]/"tmp/dot-auto-20261125")
from controller import ExperimentalReserve, first_window, guarded, route_move
from maze.environment import Environment
from maze.interface import Move
from maze.levels import Level
from maze.pilot_automaton import Mode


def corridor(row, fuel, health):
    wall = "#"*len(row)
    return Level.parse("\n".join(["Коридор",f"{fuel} {health}",wall,row,wall]))


class RuleTests(unittest.TestCase):
    def test_guard_keeps_only_path_with_exact_fuel_to_finish(self):
        level = corridor("#B..A...#",3,100)
        self.assertEqual(guarded(Environment(level).view,level.goal,(),Move.EAST),Move.WEST)

    def test_both_orders_are_considered(self):
        for row,f,h in [("#.F..A..R.B#",4,9),("#.R..A..F.B#",9,4)]:
            level = corridor(row,f,h)
            view = replace(Environment(level).view,memory=level.grid)
            self.assertEqual(route_move(view,level.goal,Move.EAST,Mode.GOAL.value),Move.WEST)

    def test_probe_returns_on_next_observation(self):
        level = Level.parse("\n".join(["Проба","30 80","###########","#........B#","#.......F.#",
                                      "#.........#","#....A....#","#.........#","#R........#","###########"]))
        env, agent = Environment(level), ExperimentalReserve("S")
        for step in (Move.EAST,Move.WEST):
            agent.choose(env.view)
            env.step(step)
        move = agent.choose(env.view)
        self.assertTrue(agent.return_probe)
        self.assertEqual(move, Move.NE)
        self.assertEqual(agent.choose(env.step(move)),Move.SW)

    def test_stagnation_window_before_any_service(self):
        for row,expected in [("#FA....B#",Move.EAST),("#.AF...B#",None)]:
            level = corridor(row,20,20)
            partial = (level.grid[0],level.grid[1][:5]+"?"+level.grid[1][6:],level.grid[2])
            self.assertEqual(first_window(replace(Environment(level).view,memory=partial)),expected)


if __name__ == "__main__":
    unittest.main()
