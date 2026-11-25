import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from maze import learning
from maze.datasets import DATA, development_sets, geometry, load_split
from maze.environment import Environment
from maze.grid import destination, locations, wave
from maze.interface import Move
from maze.levels import Level
from maze.sarsa import ALPHA, GAMMA, Sarsa, _layout, encode, train_episode


def level():
    return Level.parse("Мини\n90 90\n######\n#A..B#\n#....#\n######")


class SarsaTests(unittest.TestCase):
    def test_splits_and_required_detours(self):
        tr, val = development_sets()
        self.assertEqual((len(tr), len(val)), (90, 36))
        self.assertEqual(geometry((".##", "..#")), geometry(("..", ".#", "##")))
        for c in tr + val:
            l = c.level
            start = {p: d for p, d, _ in wave(l.grid, [l.start])}
            end = {p: d for p, d, _ in wave(l.grid, [l.goal])}
            length = start[l.goal]
            self.assertEqual(sum(stock < length for stock in (l.fuel, l.health)), c.stage-1)
            sites = locations(l.grid, "F") + locations(l.grid, "R")
            self.assertTrue(all(start[p]+end[p] > length for p in sites))
            if c.stage == 3:
                self.assertGreaterEqual(min(Counter(l.grid[y][x] for x, y in sites).values()), 2)
        with self.assertRaises(ValueError):
            load_split("test")

    def test_bit_search_matches_reference_on_partial_map(self):
        grid = ("??###??", "?.F.B??", "??.#..?", "?#...R?", "????#??")
        goal = (4, 1)
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile in "#?":
                    continue
                pos = (x, y)
                legal, *fields = _layout(grid, goal, pos)
                for targets, pair in zip(([goal], locations(grid, "F"), locations(grid, "R")), fields):
                    for ref, (length, moves) in zip((tuple(r.replace("?", ".") for r in grid), grid), pair):
                        dist = {p: d for p, d, _ in wave(ref, targets)}
                        self.assertEqual(dist.get(pos), length)
                        expected = {m for m in legal if length and
                                    dist.get(destination(grid, pos, m)) == length-1}
                        self.assertEqual(moves, expected)

    def test_same_window_different_remembered_road(self):
        env = Environment(level())
        view = replace(env.view, position=(3, 3), dx=0, dy=-2,
                       window=("###", "...", "###"))
        rows = ("#######", "#..B..#", "#.###.#", "#.....#", "#######")
        a = replace(view, memory=rows[:2] + ("#.#####",) + rows[3:])
        b = replace(view, memory=rows[:2] + ("#####.#",) + rows[3:])
        q = Sarsa()
        q.values[4] = 1
        self.assertEqual((q.choose(a), q.choose(b)), (Move.WEST, Move.EAST))

    def test_on_policy_update_executes_sampled_action_and_stops_bootstrap(self):
        env = Environment(level())
        policy = Sarsa()
        policy.values[0], policy.values[4] = 91.0, 2.0
        # Первый ход к цели, второй намеренно назад, затем вперёд до терминала.
        actions = iter((Move.EAST, Move.WEST, Move.EAST, Move.EAST, Move.EAST))
        def select(choice, *_):
            move = next(actions)
            return next(opt for opt in choice.options if opt[0] == move)
        start = encode(env.view)
        after = encode(env.step(Move.EAST))
        key = dict(start.options)[Move.EAST]
        next_key = dict(after.options)[Move.WEST]
        policy.values[key], policy.values[next_key] = 8.0, 1.0
        reward = -1 + GAMMA*after.potential - start.potential
        first = 8 + ALPHA*(reward + GAMMA*1 - 8)
        updates = []
        update = policy.update
        def record(k, r, n):
            update(k, r, n)
            updates.append((policy.values[k], n))
        with patch.object(policy, "select", side_effect=select), patch.object(policy, "update", side_effect=record):
            train_episode(env, policy, 0, None)
        self.assertEqual(env.steps, 5)
        self.assertAlmostEqual(updates[0][0], first)
        self.assertIsNone(updates[-1][1])

    def test_evaluation_freezes_q(self):
        q = Sarsa()
        before = q.values.copy()
        learning.evaluate(q, [learning.Case("пример", 1, level())])
        self.assertEqual(q.values, before)

    def test_test_gate_and_single_use_on_temporary_maps(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            levels = root / "levels"
            levels.mkdir()
            manifest = {}
            for i in range(10):
                p = levels / f"{i}.txt"
                p.write_text("Мини\n90 90\n#####\n#AB.#\n#####", encoding="utf-8")
                manifest[p.stem] = {"sha256": learning.digest(p)}
            learning.save(root / "test-manifest.json", manifest)
            for split in ("train", "val"):
                (root / f"{split}.jsonl").write_bytes((DATA / f"{split}.jsonl").read_bytes())
            rows = [dict(map=c.name, stage=c.stage, outcome="SUCCESS", steps=2) for c in load_split("val")]
            for row in rows[:4]:
                row["outcome"] = "EMPTY"
            report = dict(selected=12, validation=[dict(epoch=e, maps=rows) for e in (12, 24, 36, 48)],
                          datasets={s: learning.digest(root / f"{s}.jsonl") for s in ("train", "val")},
                          execution_sources=learning.execution_sources())
            policy = root / "policy.json"
            learning.save(policy, dict(epoch=12, values=Sarsa().values))
            report["policy_sha256"] = learning.digest(policy)
            learning.save(root / "training.json", report)
            with (patch.object(learning, "RESULTS", root), patch.object(learning, "DATA", root),
                  patch.object(learning, "LEVELS", levels),
                  patch.object(learning, "evaluate", wraps=learning.evaluate) as evaluate):
                with self.assertRaises(ValueError):
                    learning.final_test(root)
                evaluate.assert_not_called()
                rows[0]["outcome"] = "SUCCESS"
                learning.save(root / "training.json", report)
                learning.final_test(root)
                learning.final_test(root)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(len(json.loads((root / "test.json").read_text(encoding="utf-8"))["maps"]), 10)


if __name__ == "__main__":
    unittest.main()
