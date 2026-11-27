"""Сверка PPO и парный контроль правила выбора хода на двух архивных наборах."""
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
import gzip
import json
import random
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from maze.environment import Status
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import MOVES, encode, immediate
from neural_rl.rollout import MapPool, evaluate
from neural_rl.stress_hundreds import maps
from neural_rl.training import gae


def check_gae():
    # Три параллельные поездки с разными концами эпизодов, включая обрыв сбора.
    generator = torch.Generator().manual_seed(11)
    rewards, values = [torch.randn(16, 3, generator=generator, dtype=torch.float64) for _ in range(2)]
    final = torch.randn(3, generator=generator, dtype=torch.float64)
    discounts = torch.full_like(rewards, .97)
    discounts[[3, 11], 0] = 0
    discounts[[0, 15], 1] = 0
    result = gae(rewards, discounts, values, final, .95)
    independent = torch.zeros_like(result)
    for start, lane in product(range(16), range(3)):
        weight = 1.
        for t in range(start, 16):
            next_value = final[lane] if t == 15 else values[t + 1, lane]
            delta = rewards[t, lane] + discounts[t, lane] * next_value - values[t, lane]
            independent[start, lane] += weight * delta
            if discounts[t, lane] == 0:
                break
            weight *= discounts[t, lane].item() * .95
    error = (independent - result).abs().max().item()
    assert error < 1e-12
    return error


def training():
    runs = [read(p) for p in sorted((ROOT / "results/E-0012/runs").glob("ppo-*/run.json"))]
    assert len(runs) == 50
    per_run = []
    stages = []
    for r in runs:
        assert r["train_steps"] == 73728 and r["updates"] == 1536
        assert all(s["steps"] == 24576 and s["updates"] == 512 for s in r["training"])
        per_run.append(dict(id=r["config"]["id"], wins=[s["outcomes"].get("SUCCESS", 0) for s in r["training"]],
                            before=r["initial"]["wins"], after=r["screen"]["wins"]))
    for stage in range(3):
        counts = Counter()
        for r in runs:
            counts.update(r["training"][stage]["outcomes"])
        stages.append(dict(outcomes=counts, finished=sum(counts.values()),
                           runs_zero=sum(r["wins"][stage] == 0 for r in per_run)))
    return dict(stages=stages, by_run=per_run, updates_each=1536, steps_each=73728,
                before_mean=statistics.mean(r["before"] for r in per_run),
                after_mean=statistics.mean(r["after"] for r in per_run),
                gained=sum(r["after"] > r["before"] for r in per_run),
                unchanged=sum(r["after"] == r["before"] for r in per_run),
                lost=sum(r["after"] < r["before"] for r in per_run))


@torch.inference_mode()
def sample(net, pool, mode):
    tasks = iter(product(range(3), pool.cases))
    active, rows = [], []
    while True:
        while len(active) < 64 and (item := next(tasks, None)) is not None:
            repeat, case = item
            key = f"E-0014-PPO|{case.name}|{repeat}"
            rng = random.Random(int.from_bytes(sha256(key.encode()).digest(), "big"))
            active.append((case, pool.start(case), repeat, rng))
        if not active:
            return rows
        if mode == "categorical":
            data = batch([encode(env.view, False) for _, env, _, _ in active])
            probabilities = net(data)[0].softmax(-1).cpu().numpy().astype(np.float64)
        else:
            probabilities = [immediate(env.view.window) for _, env, _, _ in active]
        ongoing = []
        for item, probs in zip(active, probabilities):
            case, env, repeat, rng = item
            cdf = np.cumsum(probs)
            action = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
            assert immediate(env.view.window)[action]
            env.step(MOVES[action])
            if env.status == Status.RUNNING:
                ongoing.append(item)
            else:
                rows.append(dict(map=case.name, repeat=repeat, outcome=env.status.name,
                                 steps=env.steps, refuels=env.refuels, repairs=env.repairs))
        active = ongoing


def summarize(rows):
    return dict(episodes=len(rows), wins=sum(r["outcome"] == "SUCCESS" for r in rows),
                unique_wins=len({r["map"] for r in rows if r["outcome"] == "SUCCESS"}),
                outcomes=Counter(r["outcome"] for r in rows),
                any_service=sum(bool(r["refuels"] or r["repairs"]) for r in rows),
                both_services=sum(bool(r["refuels"] and r["repairs"]) for r in rows))


def main():
    model_id = read(ROOT / "results/E-0012/frozen.json")["ppo"]["id"]
    folder = ROOT / "results/E-0012/runs" / model_id
    config = read(folder / "run.json")["config"]
    model_hash = digest(folder / "weights.pt")
    assert model_hash == read(ROOT / "results/E-0012/frozen.json")["ppo"]["sha256"]
    setup(config["seed"])
    net = Network(config).cuda().eval()
    net.load_state_dict(torch.load(folder / "weights.pt", map_location="cuda", weights_only=True))
    summary = dict(model=model_id, weights_sha256=model_hash, gae_max_error=check_gae(), training=training(), suites={})
    stress = MapPool([case for _, case in maps()])
    with gzip.open(ROOT / "results/E-0012/evaluation/test_id" / model_id / "00000.json.gz", "rt", encoding="utf8") as f:
        archive_id = json.load(f)
    archive_stress = read(ROOT / "results/E-0013/models" / (model_id + ".json"))
    assert archive_stress["sha256"] == archive_id["weights_sha256"] == model_hash
    for suite, pool, archive in [("stress", stress, archive_stress), ("test_id_first100", Pool("test_id", 100), archive_id)]:
        greedy = [row for i in range(0, len(pool.cases), 50)
                  for row in evaluate(net, Chunk(pool, pool.cases[i:i+50]), 1)]
        keys = ["map", "outcome", "steps", "refuels", "repairs"]
        compact = lambda rows: sorted(tuple(row[k] for k in keys) for row in rows)
        assert compact(greedy) == compact(archive["rows"])
        study = {}
        for mode in ("greedy", "categorical", "uniform"):
            rows = greedy if mode == "greedy" else sample(net, pool, mode)
            assert len(rows) == len(pool.cases) * (1 if mode == "greedy" else 3)
            with gzip.open(OUT / f"{suite}-{mode}.json.gz", "wt", encoding="utf8") as f:
                json.dump(rows, f, separators=(",", ":"))
            part = study[mode] = summarize(rows)
            if mode != "greedy":
                part["by_repeat"] = [summarize([r for r in rows if r["repeat"] == i]) for i in range(3)]
            print(suite, mode, part["wins"], part["episodes"], flush=True)
        summary["suites"][suite] = study
    save(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
