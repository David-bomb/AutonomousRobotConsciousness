"""Проверки на искусственных картах. --full прогоняет всю цепочку 3×100."""
import argparse
from dataclasses import fields, replace
from importlib import import_module
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate import OBS_FIELDS, checked_lock, episode, lock_data, summarize
from maze.interface import Move, Observation
from policy_api import GreedyController, ORDER
from suite import (ARCHIVE, HERE, Case, Level, SPLITS, digest, file_hash,
                   geometry, inventory, object_hash, read, transfer, write_new)

# Фиктивные веса кодируют один постоянный ход независимо от карты.
ADAPTER = b"""
from maze.interface import Move, Observation
from policy_api import GreedyController, ORDER

def load(checkpoint):
    action = Move[checkpoint.decode("ascii")]
    def scores(view):
        assert type(view) is Observation
        assert isinstance(view.memory, tuple) and isinstance(view.position, tuple)
        if view.last_move is None:
            assert view.moved is None and "?" in "".join(view.memory)
        return [int(m is action) for m in ORDER]
    return GreedyController(scores)
"""


def toy(n=0, stocks=(1, 2)):
    # Асимметричная рамка и девять изменяемых клеток дают разные геометрии D4.
    cells = "".join("#" if (n >> bit) & 1 else "." for bit in range(9))
    grid = ["################", "#AFRB..........#", "#.##...........#",
            "#." + cells + "....#", "################"]
    name = f"toy-{n:03}"
    src = "\n".join([name, f"{stocks[0]} {stocks[1]}", *grid])
    case = Case("I", name, geometry(grid), object_hash([grid, *stocks]), Level.parse(src))
    return case, dict(name=name, grid=grid, fuel=stocks[0], health=stocks[1])


def files(root):
    paths = []
    for i, split in enumerate(SPLITS):
        path = root / f"{split}.jsonl"
        paths.append(path)
        with path.open("w", encoding="utf-8") as stream:
            for n in range(i*100, (i+1)*100):
                stream.write(json.dumps(toy(n)[1]) + "\n")
    return paths


class Tests(unittest.TestCase):
    def test_contract_and_outcomes(self):
        self.assertEqual(set(OBS_FIELDS) | {"last_move"}, {f.name for f in fields(Observation)})
        c, _ = toy()
        row = episode(c, ADAPTER, b"EAST")
        self.assertEqual((row["outcome"], row["steps"], row["refuels"], row["repairs"]),
                         ("SUCCESS", 3, 1, 1))
        self.assertEqual(row["actions_sha256"], digest(bytes([1, 1, 1])))
        self.assertEqual(row, episode(c, ADAPTER, b"EAST"))
        self.assertEqual(episode(c, ADAPTER, b"NORTH")["outcome"], "EMPTY")
        self.assertEqual(episode(toy(stocks=(2, 1))[0], ADAPTER, b"NORTH")["outcome"], "BROKEN")
        cycle = (b"from maze.interface import Move\nclass C:\n def choose(self, v):\n"
                 b"  return Move.WEST if v.position[0] == 3 else Move.EAST\ndef load(w): return C()")
        row = episode(c, cycle, b"")
        self.assertEqual((row["outcome"], row["steps"]), ("LIMIT", 8*5*16))
        self.assertEqual(row["refuels"] + row["repairs"], row["steps"])

    def test_tie(self):
        self.assertEqual(tuple(Move), ORDER)
        self.assertIs(GreedyController(lambda _: [0]*8).choose(None), Move.NORTH)
        self.assertIs(GreedyController(lambda _: [-float("inf"), 1, 1, 0, 0, 0, 1, 0]).choose(None), Move.EAST)
        for values in ([0]*7, [-float("inf")]*8, [float("nan")]*8):
            with self.assertRaises(ValueError):
                GreedyController(lambda _: values).choose(None)

    def test_threshold_and_integrity(self):
        cases = [replace(toy(i)[0], split=SPLITS[i//100]) for i in range(300)]
        weights = digest(b"EAST")
        rows = [dict(split=c.split, id=c.name, scenario_sha256=c.scenario, geometry=c.geometry,
                     checkpoint_sha256=weights, actions_sha256=digest(bytes([1]*3)),
                     steps=3, refuels=1, repairs=1, outcome="SUCCESS" if i % 100 < 80 else "EMPTY")
                for i, c in enumerate(cases)]
        self.assertTrue(summarize(cases, rows, weights)["passed"])
        rows[0]["outcome"] = "EMPTY"
        rows[299]["outcome"] = "SUCCESS"
        self.assertEqual(summarize(cases, rows, weights)["successes"], {"I": 79, "II": 80, "III": 81})
        self.assertFalse(summarize(cases, rows, weights)["passed"])
        for bad in (rows[:-1], [rows[1], *rows[1:]], [dict(rows[0], checkpoint_sha256="0"*64), *rows[1:]]):
            with self.assertRaises(ValueError):
                summarize(cases, bad, weights)

    def test_geometry_and_stocks(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            paths = files(Path(temp))
            _, spec = inventory(paths)
            lines = paths[0].read_text().splitlines()
            first = json.loads(lines[0])
            first["health"] += 1
            lines[0] = json.dumps(first)
            paths[0].write_text("\n".join(lines)+"\n")
            _, changed = inventory(paths)
            self.assertEqual(spec["sets"]["I"]["geometries"], changed["sets"]["I"]["geometries"])
            for key in ("stocks", "scenarios", "sha256_lf"):
                self.assertNotEqual(spec["sets"]["I"][key], changed["sets"]["I"][key])
            first["name"] = json.loads(lines[1])["name"]
            first["grid"] = [s[::-1] for s in first["grid"]]
            lines[1] = json.dumps(first)
            paths[0].write_text("\n".join(lines)+"\n")
            with self.assertRaisesRegex(ValueError, "Повтор геометрии"):
                inventory(paths)

    def test_checkpoint_and_adapter(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            w, a, lock = root / "weights", root / "adapter.py", root / "lock.json"
            w.write_bytes(b"EAST")
            a.write_bytes(ADAPTER)
            info = lock_data(a, w, file_hash(w), ARCHIVE, [])
            write_new(lock, info)
            checked_lock(lock, object_hash(info))
            with self.assertRaises(ValueError):
                checked_lock(lock, "0"*64)
            a.write_bytes(ADAPTER + b"\n")
            with self.assertRaises(ValueError):
                checked_lock(lock, object_hash(info))
            w.write_bytes(b"WEST")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                checked_lock(lock, object_hash(info))


def command(*args):
    return subprocess.run([sys.executable, "-X", "utf8", "-B", str(HERE / "evaluate.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8")


def full():
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        root = Path(temp)
        maps = files(root)
        exposed = root / "seen.txt"
        exposed.write_text(toy(0)[0].geometry + "\n")
        with unittest.TestCase().assertRaisesRegex(ValueError, "пересекаются"):
            transfer(maps, [exposed])
        exposed.write_text(toy(500)[0].geometry + "\n")
        spec, weights, adapter = root/"spec.json", root/"weights", root/"adapter.py"
        lock, out = root/"lock.json", root/"run"
        write_new(spec, transfer(maps, [exposed]))
        weights.write_bytes(b"EAST")
        adapter.write_bytes(ADAPTER)
        proc = command("freeze", "--checkpoint", weights, "--adapter", adapter, "--sha256", file_hash(weights),
                       "--spec", spec, "--out", lock)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        args = ("--lock", lock, "--lock-sha256", object_hash(read(lock)), "--out", out)
        for name in ("run", "verify"):
            proc = command(name, *args)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
        result = read(out/"summary.json")
        unittest.TestCase().assertEqual(result["successes"], {f"transfer-{i}": 100 for i in range(1, 4)})
        log = out/"episodes.jsonl"
        log.write_bytes(log.read_bytes().replace(b'"SUCCESS"', b'"EMPTY"', 1))
        unittest.TestCase().assertNotEqual(command("verify", *args).returncode, 0)
        print("OK: 300 искусственных карт, 900 физических ходов, подмена журнала обнаружена.")


def adapter_smoke():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    net = import_module("results.E-0016.policy").Policy().cpu()
    zero = {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    checkpoint = BytesIO()
    torch.save(zero, checkpoint)
    level = Level.parse("toy-policy\n1 1\n###########\n#AB.F.R...#\n#..F..R...#\n###########")
    case = Case("toy", "toy-policy", geometry(level.grid), "toy", level)
    row = episode(case, (HERE / "e16_adapter.py").read_bytes(), checkpoint.getvalue())
    unittest.TestCase().assertEqual((row["outcome"], row["steps"], row["actions_sha256"]),
                                    ("SUCCESS", 1, digest(b"\x01")))
    print("OK: нулевые веса E-0016, искусственная карта, физический ход EAST.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--adapter-smoke", action="store_true")
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        sys.exit(1)
    if args.full:
        full()
    if args.adapter_smoke:
        adapter_smoke()
