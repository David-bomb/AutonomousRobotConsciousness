"""Доработка копий четырёх отобранных сетей на учебных примерах диагностических задач."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

from neural_rl.experiment import digest, read, save
from .diagnostic_eval import DiagnosticPool
from .diagnostics import FOLDER
from .final_eval import freeze_file
from .learn import DirectNetwork, FAMILIES, evaluate_rows, setup
from .report import NAMES, table, write_text
from .session import OUT, ROOT, check_time, locked, log


RECIPE = FOLDER / "training-recipe.json"


def register():
    banks = read(FOLDER / "verified.json")
    spec = dict(source_sha256=digest(Path(__file__)), banks=banks["sha256"], families=FAMILIES,
                steps=73728, selection="one validation-selected checkpoint per family",
                stage="1 + within-group case number modulo 3", evaluation="1000 independent controls, once",
                optimizer="fresh Adam, same hyperparameters", seed_offset=1000000)
    # JSON сохраняет список семейств, даже если константа в Python — кортеж.
    spec["families"] = list(FAMILIES)
    first = not RECIPE.exists()
    freeze_file(RECIPE, spec)
    if first:
        log("diagnostic_training_registered", steps=spec["steps"], families=spec["families"],
            train=1000, control=1000, source_sha256=spec["source_sha256"])


def bank(which, spec):
    path = FOLDER / f"{which}.jsonl.gz"
    if digest(path) != spec["banks"][which]:
        raise ValueError("Изменилась выборка диагностических примеров")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(r["proof"]["status"] != "SUCCESS" or r["bank"] != which for r in rows):
        raise ValueError("Запись не прошла допуск")
    return DiagnosticPool(rows)


def train_copy(family):
    import numpy as np
    import torch
    from neural_rl.training import train_ppo, train_q

    spec = read(RECIPE)
    base = read(OUT / "frozen.json")[family]
    original = OUT / "runs" / base["id"]
    target = FOLDER / "training" / family
    target.mkdir(parents=True, exist_ok=True)
    if digest(original / "weights.pt") != base["sha256"]:
        raise ValueError("Изменились исходные веса")
    if (target / "run.json").exists():
        result = read(target / "run.json")
        if result["base"] != base or digest(target / "weights.pt") != result["sha256"]:
            raise ValueError("Изменились сохранённые веса")
        return
    config = read(original / "run.json")["config"]
    seed = config["seed"] + spec["seed_offset"]
    setup(seed)
    pool = bank("train", spec)
    pool.cases = [replace(c, stage=1 + int(c.name.rsplit("-", 1)[1]) % 3) for c in pool.cases]
    groups = {c.name: int(c.name.split("-")[2]) for c in pool.cases}
    net = DirectNetwork(config).cuda()
    net.load_state_dict(torch.load(original / "weights.pt", weights_only=True, map_location="cuda"))
    initial = [p.detach().clone() for p in net.parameters()]
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
    log("diagnostic_training_started", id=base["id"], seed=seed)
    rng = random.Random(seed)
    if family == "ppo":
        stages = train_ppo(net, opt, config, pool, rng)
    else:
        stages = train_q(net, opt, config, pool, rng, np.random.default_rng(seed))
    seen = {name for stage in stages for name in stage["maps"]}
    coverage = Counter(groups[name] for name in seen)
    steps = sum(stage["steps"] for stage in stages)
    delta = max((p.detach() - old).abs().max().item() for p, old in zip(net.parameters(), initial))
    if (steps != spec["steps"] or set(coverage) != set(range(1, 21)) or not delta > 0
            or not math.isfinite(delta) or any(not math.isfinite(s["mean_loss"]) or not s["updates"] for s in stages)):
        raise ValueError("Доработка весов или охват диагностических групп не подтверждены")
    temp = target / "weights.pending"
    torch.save(net.state_dict(), temp)
    temp.replace(target / "weights.pt")
    result = dict(base=base, sha256=digest(target / "weights.pt"), seed=seed, steps=steps,
                  parameter_delta=delta, training=stages, coverage=dict(coverage), distinct_cases=len(seen))
    save(target / "run.json", result)
    log("diagnostic_training_completed", id=base["id"], steps=steps, cases=len(seen), groups=len(coverage))


def launch(family):
    path = OUT / "process-logs" / f"diag-train-{family}.txt"
    with path.open("ab") as out:
        return subprocess.call([sys.executable, "-B", "-u", "-m", "neural_rl.corpus_study.diagnostic_training",
                                "train", family], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def evaluate_copies():
    import torch
    spec = read(RECIPE)
    control = bank("control", spec)
    summaries = []
    for family in FAMILIES:
        folder = FOLDER / "training" / family
        result = read(folder / "run.json")
        base = result["base"]["id"]
        config = read(OUT / "runs" / base / "run.json")["config"]
        if digest(folder / "weights.pt") != result["sha256"]:
            raise ValueError("Изменились веса после обучения")
        setup(result["seed"])
        net = DirectNetwork(config).cuda()
        net.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cuda"))
        rows = evaluate_rows(net, control, chunk=48)
        before = read(FOLDER / f"score-{family}.json")["groups"]
        old = {r["map"]: r["outcome"] for g in before.values() for r in g["rows"]}
        if len(rows) != 1000 or {r["map"] for r in rows} != set(old):
            raise ValueError("Не совпадают диагностические контрольные случаи")
        gains = sum(old[r["map"]] != "SUCCESS" and r["outcome"] == "SUCCESS" for r in rows)
        losses = sum(old[r["map"]] == "SUCCESS" and r["outcome"] != "SUCCESS" for r in rows)
        wins = sum(s == "SUCCESS" for s in old.values())
        save(folder / "control.json", dict(id=base, weights_sha256=result["sha256"], gains=gains, losses=losses, rows=rows))
        summaries.append([NAMES[family], result["distinct_cases"], wins, wins + gains - losses, gains, losses])
        log("diagnostic_trained_control", id=base, before=wins, after=wins+gains-losses)
    write_text(FOLDER / "training-results.md", "## Обучение на независимых примерах двадцати умений\n\n"
               "Копия каждой из четырёх выбранных сетей получила ещё 73 728 физических шагов на учебном "
               "диагностическом банке. Контрольные 1 000 сценариев общие для сравнения. "
               "Результаты исходных моделей на основных выборках приведены отдельно.\n\n" +
               table(["Семейство", "Охвачено учебных из 1000", "До", "После", "Новых финишей", "Потерянных финишей"], summaries) + "\n")


def follow():
    register()
    while not (OUT / "diagnostic-results.md").exists():
        check_time()
        time.sleep(20)
    with ThreadPoolExecutor(2) as workers:
        codes = list(workers.map(launch, FAMILIES))
    for family, code in zip(FAMILIES, codes):
        if code:
            check_time()
            log("diagnostic_training_retry", family=family, parallel=1)
            if launch(family):
                raise RuntimeError(f"Не завершено диагностическое обучение {family}")
    evaluate_copies()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "train", "follow"))
    parser.add_argument("family", nargs="?", choices=FAMILIES)
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "train":
        train_copy(args.family)
    else:
        with locked(FOLDER / ".training.lock"):
            follow()
