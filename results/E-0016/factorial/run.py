"""Запуск из корня: python -B -m results.E-0016.factorial.run register|technical|full."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
from importlib import import_module
import json
import math
import random
import time

import numpy as np
import torch

from maze.environment import Status
from neural_rl import perception, rollout, training
from neural_rl.corpus_study import learn
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network
from .inputs import INDEX, OUT, REG, ROOT, SNAPSHOT, register, weights_hash

formula = import_module("results.E-0016.reward_test")
old_reward = rollout.reward
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")


def batch_cpu(states):
    return tuple(torch.from_numpy(np.stack(column)) for column in zip(*states))


class CpuTorch:
    # У неизменённого q_update только arange и as_tensor привязаны к cuda.
    def __getattr__(self, name):
        return getattr(torch, name)

    def arange(self, *args, device):
        assert device == "cuda"
        return torch.arange(*args, device="cpu")

    def as_tensor(self, values, *, dtype, device):
        assert device == "cuda"
        return torch.as_tensor(values, dtype=dtype, device="cpu")


class RecordedStage(rollout.Stage):
    def __init__(self, *args):
        super().__init__(*args)
        self.finished = defaultdict(Counter)

    def advance(self, lane, action, remaining, gamma):
        env = self.environments[lane]
        result = super().advance(lane, action, remaining, gamma)
        if env.status != Status.RUNNING:
            self.finished[env.level.title][env.status.name] += 1
        return result

    def report(self):
        return dict(**super().report(), finished=self.finished)


def reward_one(before, after, status):
    return formula.shaped(before, after, status, .97)


@torch.inference_mode()
def validate(net, dev, index):
    net.eval()
    rows = []
    for start in range(0, len(dev.cases), 32):
        cases = dev.cases[start:start+32]
        active = []
        for case in cases:
            env = dev.start(case)
            meta = index[case.name]
            assert (env.steps, env.position, env.fuel, env.health) == (
                0, env.level.start, meta["fuel"], meta["health"])
            active.append((case, env, {}, formula.potential(env.view), 0., 0.))
        while active:
            # Кеш оценщика запоминает лишь результат той же замороженной функции.
            # Снимок наблюдения служит полным ключом. Каждый физический ход выполняется.
            fresh = [(env.view, cache) for _, env, cache, *_ in active if env.view not in cache]
            if fresh:
                states = [perception.encode(view, False) for view, _ in fresh]
                actions = net(batch_cpu(states))[0].argmax(1).tolist()
                for (view, cache), action, state in zip(fresh, actions, states):
                    assert state.legal[action]
                    cache[view] = action
            remaining = []
            for case, env, cache, initial_phi, r0, r1 in active:
                before = env.view
                factor = .97**env.steps
                after = env.step(perception.MOVES[cache[before]])
                assert env.moved
                r0 += factor * old_reward(before, after, env.status)
                r1 += factor * reward_one(before, after, env.status)
                if env.status == Status.RUNNING:
                    remaining.append((case, env, cache, initial_phi, r0, r1))
                    continue
                terminal = 8 if env.status == Status.SUCCESS else -8
                exact = -.12*(1-.97**env.steps)/.03 + terminal*factor - initial_phi
                assert abs(exact - r1) < 1e-9
                rows.append(dict(map=case.name, **index[case.name], outcome=env.status.name,
                                 steps=env.steps, refuels=env.refuels, repairs=env.repairs,
                                 return_R0=r0, return_R1=r1, residual_R1=abs(r1-exact)))
            active = remaining
        # Сверка всех поездок с штатным оценщиком без этого кеша.
        other = rollout.evaluate(net, learn.Chunk(dev, cases), 1)
        ours = sorted(rows[-len(cases):], key=lambda r:r["map"])
        for r, ref in zip(ours, other, strict=True):
            assert all(r[k] == v for k, v in ref.items()), (r, ref)
    return sorted(rows, key=lambda r:r["map"])


def group_pool(pool, arm, index):
    if arm.endswith("C1"):
        ordered = sorted(pool.cases, key=lambda c: (
            index[c.name]["deficits"], index[c.name]["direct"], c.name))
        pool.cases = [replace(c, stage=index[c.name]["c1"]) for c in ordered]


def run_one(phase, arm, base, reg, index):
    cfg = dict(base, ticks=256 if phase == "technical" else 1024)
    seed = cfg["seed"]
    folder = OUT / phase / str(seed) / arm
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "training.json"
    weights = folder / "weights.pt"
    val = folder / "validation.json"
    finish = folder / "verified.json"
    if finish.exists():
        proof = read(finish)
        assert proof["registration"] == digest(REG)
        assert all(digest(folder/k) == v for k, v in proof["files"].items())
        return
    start = time.perf_counter()
    torch.manual_seed(seed)
    net = Network(cfg)
    initial = weights_hash(net)
    assert initial == reg["initial"][str(seed)]
    assert cfg["gamma"] == .97 and cfg["family"] == "double" and cfg["horizon"] == 1
    if not record.exists():
        pool = learn.Pool("train", cfg["geometry_count"])
        group_pool(pool, arm, index)
        rollout.reward = old_reward if arm.startswith("R0") else reward_one
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        history = training.train_q(net.train(), optimizer, cfg, pool, random.Random(seed),
                                   np.random.default_rng(seed))
        pool.db.close()
        for stage in history:
            assert stage["steps"] == stage["decisions"] == cfg["lanes"]*cfg["ticks"]
            assert stage["updates"] == cfg["ticks"]//2 and math.isfinite(stage["mean_loss"])
        assert all(p.device.type == "cpu" and torch.isfinite(p).all() for p in net.parameters())
        assert weights_hash(net) != initial
        for state in optimizer.state.values():
            assert state["step"].item() == 3*cfg["ticks"]//2
            assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
        pending = folder / "pending.pt"
        torch.save(net.state_dict(), pending)
        pending.replace(weights)
        save(record, dict(config=cfg, arm=arm, initial=initial, final=weights_hash(net),
                          weights_sha256=digest(weights), train_seconds=time.perf_counter()-start,
                          frozen_utc=datetime.now(timezone.utc).isoformat(), history=history))
    saved = read(record)
    assert saved["initial"] == initial and saved["config"] == cfg and saved["arm"] == arm
    assert saved["weights_sha256"] == digest(weights)
    net.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    net.requires_grad_(False)
    assert weights_hash(net) == saved["final"]
    dev = learn.Pool("validation_id", 60)
    vstart = time.perf_counter()
    rows = validate(net, dev, index)
    assert weights_hash(net) == saved["final"]
    assert len(rows) == len({r["map"] for r in rows}) == 240
    dev.db.close()
    save(val, rows)
    save(finish, dict(registration=digest(REG), files={p.name:digest(p) for p in (weights, record, val)},
                      evaluation_seconds=time.perf_counter()-vstart, reference_rows=len(rows),
                      cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      cuda_initialized=torch.cuda.is_initialized(), final_weights_unchanged=True))
    assert not torch.cuda.is_initialized()
    print(phase, seed, arm, sum(r["outcome"] == "SUCCESS" for r in rows), "/240;",
          round(saved["train_seconds"],1), "s train;", round(time.perf_counter()-vstart,1),
          "s eval", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register","technical","full"))
    phase = parser.parse_args().phase
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if phase == "register":
        register()
        return
    reg = read(REG)
    for path, h in {**reg["base_sources"], **reg["sources"]}.items():
        if path != "results/E-0016/protocol.md":  # Общий протокол может дополняться другим чатом.
            assert digest(ROOT / path) == h, path
    assert reg["selected_db_sha256"] == digest(SNAPSHOT) and reg["index_gz_sha256"] == digest(INDEX)
    index = json.loads(gzip.decompress(INDEX.read_bytes()))
    learn.DB = SNAPSHOT
    perception.SIDE = reg["input_canvas"]
    training.torch = CpuTorch()
    training.batch = rollout.batch = batch_cpu
    training.Stage = RecordedStage
    if phase == "full":
        gate = read(OUT / "technical" / "gate.json")
        assert gate["passed"] and gate["registration"] == digest(REG)
    for base in reg["configs"][:1 if phase == "technical" else 5]:
        for arm in ARMS:
            run_one(phase, arm, base, reg, index)
    if phase == "technical":
        folder = OUT / phase / str(reg["configs"][0]["seed"])
        elapsed = 0.
        for arm in ARMS:
            elapsed += read(folder/arm/"training.json")["train_seconds"]
            elapsed += read(folder/arm/"verified.json")["evaluation_seconds"]
        projected = 20 * elapsed
        save(OUT/phase/"gate.json", dict(passed=projected <= 2*3600, registration=digest(REG),
                                         quartet_seconds=elapsed, projected_full_seconds=projected))
        print("projected full seconds", round(projected,1), flush=True)


if __name__ == "__main__":
    main()
