"""E-0011. Регистрация, обучение и оценка серий на CUDA."""
import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import msvcrt
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from maze.datasets import DATA, development_sets
from maze.learning import validation_score
from .networks import Network
from .rollout import MapPool, evaluate
from .training import train_ppo, train_q

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/E-0011"
FAMILIES = ("double", "dueling", "rainbow", "ppo", "hrl_astar")
SEEDS = (39139, 71809, 92221, 130031, 175003)
ARCHITECTURES = ((8, 48, 1), (12, 64, 1), (8, 64, 2), (12, 64, 2), (16, 96, 2))


def configs():
    for seed_id, seed in enumerate(SEEDS, 1):
        for variant in range(10):
            large_lr = variant % 2
            for family in FAMILIES:
                config = dict(family=family, variant=variant+1, seed=seed,
                              id=f"{family}/v{variant+1:02}-s{seed_id}",
                              architecture=list(ARCHITECTURES[variant//2]),
                              lr=(0.0004, 0.0012)[large_lr], gamma=0.97, lanes=24, ticks=256,
                              horizon=(3, 6)[large_lr] if family == "hrl_astar" else 1)
                if family == "ppo":
                    config.update(rollout=16, epochs=2, batch=96, clip=(0.15, 0.25)[large_lr],
                                  entropy=(0.015, 0.035)[large_lr], **{"lambda": 0.95})
                else:
                    config.update(batch=64, replay=8192, sync=(32, 64)[large_lr])
                yield config


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(".pending.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def registration():
    source = [*sorted(Path(__file__).parent.glob("*.py"))]
    source.extend(ROOT/"maze"/(name+".py") for name in
                  ("datasets", "interface", "environment", "memory", "grid", "levels", "learning"))
    return dict(format=1, runs=list(configs()), dataset={name: digest(DATA/name)
                for name in ("train.jsonl", "val.jsonl", "test-manifest.json")},
                sources={p.relative_to(ROOT).as_posix(): digest(p) for p in source},
                selection="Худшая группа val, число финишей, среднее число ходов на финишах, меньший номер варианта и зерна.",
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/".lock").open("a+b") as stream:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run_one(folder, config, train, val):
    start = time.perf_counter()
    seed = config["seed"]
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Network(config).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    if net.family == "ppo":
        history = train_ppo(net, optimizer, config, train, rng)
    else:
        history = train_q(net, optimizer, config, train, rng, np_rng)
    learning_seconds = time.perf_counter()-start
    if not all(math.isfinite(s["mean_loss"]) for s in history):
        raise RuntimeError("Нечисловое значение функции потерь")
    expected = config["lanes"]*config["ticks"]
    if len(history) != 3 or any(s["steps"] != expected or s["updates"] <= 0 for s in history):
        raise RuntimeError("Неполный бюджет обучения")
    rows = evaluate(net, val, config["horizon"])
    worst, wins, negative_mean = validation_score(rows)
    weights = folder/"weights.pt"
    pending = folder/"weights.pending"
    torch.save(net.state_dict(), pending)
    pending.replace(weights)
    report = dict(config=config, train=history, validation=rows, wins=wins, worst=worst,
                  mean_success_steps=-negative_mean if wins else None,
                  outcomes=dict(Counter(r["outcome"] for r in rows)), weights_sha256=digest(weights),
                  train_seconds=round(learning_seconds, 3), seconds=round(time.perf_counter()-start, 3))
    save(folder/"run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "smoke"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("Для E-0011 требуется CUDA")
    train, val = (MapPool(cases) for cases in development_sets())
    if args.action == "smoke":
        for config in list(configs())[:len(FAMILIES)]:
            config.update(lanes=8, ticks=16, batch=16)
            folder = ROOT/"tmp/E-0011-smoke"/config["id"]
            folder.mkdir(parents=True, exist_ok=True)
            result = run_one(folder, config, train, val)
            print(config["family"], result["wins"], result["seconds"], flush=True)
        return
    manifest = args.output/"manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    settings = registration()
    if args.action == "register":
        if manifest.exists() and read(manifest) != settings:
            raise RuntimeError("Папка занята другой регистрацией")
        save(manifest, settings)
        print(manifest, flush=True)
        return
    if read(manifest) != settings:
        raise RuntimeError("Код, окружение или выборки отличаются от регистрации")
    for config in settings["runs"]:
        if args.family and config["family"] != args.family:
            continue
        folder = args.output/config["id"]
        with run_lock(folder) as acquired:
            if not acquired:
                continue
            if (folder/"run.json").is_file():
                existing = read(folder/"run.json")
                if existing["config"] != config or existing["weights_sha256"] != digest(folder/"weights.pt"):
                    raise RuntimeError(f"Повреждён результат {folder}")
                continue
            report = run_one(folder, config, train, val)
            print(config["id"], f"val {report['wins']}/36", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
