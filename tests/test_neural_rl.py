"""Проверки границы наблюдения, длительности действий и численных целей E-0011."""
import unittest
from collections import deque
from dataclasses import replace
from importlib.util import find_spec


@unittest.skipUnless(find_spec("torch") and find_spec("numpy"), "Для E-0011 нужен PyTorch")
class NeuralNavigationTests(unittest.TestCase):
    def test_observation_and_corner(self):
        import numpy as np
        from maze.environment import Environment
        from maze.levels import Level
        from neural_rl.perception import encode, astar, MOVES
        level = Level.parse("Тест\n50 60\n#######\n#A.#.B#\n##..F.#\n#######")
        view = Environment(level).view
        direct = encode(view, False)
        self.assertEqual(len(direct.numbers), 29)
        self.assertEqual([MOVES[i].value for i in np.flatnonzero(direct.legal)], [(1, 0)])
        self.assertTrue(direct.image[1, 5] & 1)
        self.assertTrue(direct.image[1, 5] & 32)
        self.assertEqual(direct.image[2, 4], 12)
        self.assertEqual(direct.image[3, 4], 1)
        self.assertEqual(encode(view, True).legal.tolist(), [True, True, False, True])
        changed = replace(view, fuel=25)
        self.assertTrue(np.array_equal(direct.image, encode(changed, False).image))
        self.assertFalse(np.array_equal(direct.numbers, encode(changed, False).numbers))
        self.assertFalse(astar(view.memory, view.position, (5, 1)))
        self.assertEqual(len(astar(("#.##", "#..#", "##.#"), (1, 0), (2, 2))), 3)

    def test_nstep_discount_at_option_and_episode_boundary(self):
        from neural_rl.training import Replay, emit
        replay = Replay(4)
        queue = deque([("s0", 1, 2., 0.81, "s1"), ("s1", 2, 3., 0., "end")])
        emit(queue, replay)
        emit(queue, replay)
        self.assertEqual(replay.items, [("s0", 1, 2.+0.81*3, 0., "end"), ("s1", 2, 3., 0., "end")])

    def test_projection_and_generalized_advantage(self):
        import torch
        from neural_rl.networks import project
        from neural_rl.training import gae
        support = torch.tensor([-1., 0., 1.])
        probabilities = torch.tensor([[0., 1., 0.]]*3)
        actual = project(probabilities, torch.tensor([0., 0.25, 2.]), torch.zeros(3), support)
        torch.testing.assert_close(actual, torch.tensor([[0., 1., 0.], [0., 0.75, 0.25], [0., 0., 1.]]))
        adv = gae(torch.tensor([[1.], [2.]]), torch.tensor([[0.5], [0.]]),
                  torch.tensor([[0.], [1.]]), torch.tensor([100.]), 1.)
        torch.testing.assert_close(adv, torch.tensor([[2.], [1.]]))

    def test_a_star_obeys_physical_budget_and_terminal(self):
        from maze.environment import Environment, Status
        from maze.levels import Level
        from neural_rl.rollout import execute
        env = Environment(Level.parse("Тест\n100 100\n#######\n#A...B#\n#######"))
        # После разведки вернуться на старт с полной памятью о дороге до B.
        from maze.interface import Move
        for move in [Move.EAST]*3 + [Move.WEST]*3:
            env.step(move)
        _, duration = execute(env, 0, True, 2, 0.97)
        self.assertEqual((duration, env.position), (2, (3, 1)))
        _, duration = execute(env, 0, True, 6, 0.97)
        self.assertEqual((duration, env.status), (2, Status.SUCCESS))
