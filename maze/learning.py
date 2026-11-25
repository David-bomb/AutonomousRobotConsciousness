"""Раздельные команды обучения и однократной итоговой оценки."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from .datasets import Case, DATA, STAGES, development_sets, load_split
from .environment import Environment, Status
from .levels import LEVELS, Level
from .runner import run
from .sarsa import Sarsa, train_episode

RESULTS = Path(__file__).resolve().parents[1] / "results" / "E-0001"
SEED = 39139
EPOCHS = (12, 24, 48)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def execution_sources():
    folder = Path(__file__).parent
    return {name: digest(folder / name) for name in
            ("sarsa.py", "runner.py", "environment.py", "grid.py", "interface.py", "memory.py", "levels.py")}


def evaluate(policy, cases):
    results = []
    for case in cases:
        env = Environment(case.level)
        status = run(env, policy)
        results.append(dict(map=case.name, stage=case.stage, outcome=status.name,
                            steps=env.steps, refuels=env.refuels, repairs=env.repairs))
    return results


def validation_score(rows):
    groups = [[r for r in rows if r["stage"] == s] for s in STAGES]
    if len(rows) != 36 or any(len(g) != 12 for g in groups) or len({r["map"] for r in rows}) != 36:
        raise ValueError("Нужно по 12 отдельных карт val на ступень")
    successes = [r for r in rows if r["outcome"] == "SUCCESS"]
    worst = min(sum(r["outcome"] == "SUCCESS" for r in group) for group in groups)
    mean = sum(r["steps"] for r in successes) / len(successes) if successes else float("inf")
    return worst, len(successes), -mean


def admitted(rows):
    worst, successes, _ = validation_score(rows)
    return worst >= 9 and successes >= 29


def train(output):
    train_cases, val = development_sets()
    output.mkdir(parents=True, exist_ok=False)
    rng, policy = random.Random(SEED), Sarsa()
    envs = {c.name: Environment(c.level) for c in train_cases}
    checkpoints, history = [], []
    best = None
    started = time.perf_counter()
    for stage, epochs in zip(STAGES, EPOCHS):
        current = [c for c in train_cases if c.stage == stage]
        print(f"Ступень {stage}: {epochs} эпох", flush=True)
        for epoch in range(1, epochs+1):
            batch = current.copy()
            for earlier in range(1, stage):
                batch += rng.sample([c for c in train_cases if c.stage == earlier], 10)
            rng.shuffle(batch)
            epsilon = 0.24 - 0.20*(epoch-1)/(epochs-1)
            outcomes = Counter()
            steps = 0
            for case in batch:
                env = envs[case.name]
                train_episode(env, policy, epsilon, rng)
                outcomes[env.status.name] += 1
                steps += env.steps
            history.append(dict(stage=stage, epoch=epoch, epsilon=epsilon, outcomes=dict(outcomes), steps=steps))
            if stage == 3 and epoch % 12 == 0:
                rows = evaluate(policy, val)
                score = validation_score(rows)
                checkpoints.append(dict(epoch=epoch, maps=rows))
                print(f"Эпоха {epoch}: val {score[1]}/36, худшая группа {score[0]}/12", flush=True)
                if best is None or score > best:
                    best = score
                    save(output / "policy.json", dict(format=1, epoch=epoch, values=policy.values))
    selected = read(output / "policy.json")["epoch"]
    report = dict(seed=SEED, epochs=EPOCHS, history=history, validation=checkpoints, selected=selected,
                  policy_sha256=digest(output / "policy.json"),
                  datasets={s: digest(DATA / f"{s}.jsonl") for s in ("train", "val")},
                  execution_sources=execution_sources(),
                  seconds=round(time.perf_counter()-started, 2))
    save(output / "training.json", report)
    return admitted(next(c["maps"] for c in checkpoints if c["epoch"] == selected))


def final_test(output):
    report = read(output / "training.json")
    policy_file = output / "policy.json"
    record = read(policy_file)
    checkpoints = report["validation"]
    expected = [(c.name, c.stage) for c in load_split("val")]
    if ([ck["epoch"] for ck in checkpoints] != [12, 24, 36, 48] or
        any([(r["map"], r["stage"]) for r in ck["maps"]] != expected for ck in checkpoints)):
        raise ValueError("Неполная оценка на зарегистрированных картах val")
    winner = max(checkpoints, key=lambda ck: validation_score(ck["maps"]))
    if (digest(policy_file) != report["policy_sha256"] or
        record["epoch"] != winner["epoch"] or report["selected"] != winner["epoch"] or
        not admitted(winner["maps"])):
        raise ValueError("Таблица не соответствует выбранному снимку или не прошла порог val")
    if any(digest(DATA / f"{s}.jsonl") != report["datasets"][s] for s in ("train", "val")):
        raise ValueError("После обучения изменился состав выборок")
    if execution_sources() != report["execution_sources"]:
        raise ValueError("После фиксации модели изменился код контроллера или среды")
    # Один общий результат для опыта, в том числе при повторном обучении в другой папке.
    result_file = RESULTS / "test.json"
    if result_file.exists():
        saved = read(result_file)
        if saved["policy_sha256"] != report["policy_sha256"] or len(saved["maps"]) != 10:
            raise ValueError("Итоговая попытка уже занята другим или неполным результатом")
        print(f"Сохранена оценка этой модели на десяти уровнях: {result_file}")
        return
    manifest = read(DATA / "test-manifest.json")
    if set(manifest) != {p.stem for p in LEVELS.glob("*.txt")} or len(manifest) != 10:
        raise ValueError("Изменился состав отложенных уровней")
    cases = []
    for name, item in manifest.items():
        source = (LEVELS / f"{name}.txt").read_text(encoding="utf-8")
        if hashlib.sha256(source.encode()).hexdigest() != item["sha256"]:
            raise ValueError(f"Изменился уровень {name}")
        cases.append(Case(name, 0, Level.parse(source)))
    policy = Sarsa(record["values"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Незавершённая попытка оставляет отдельную запись и не выдаётся за итоговую.
    with result_file.with_suffix(".opened.json").open("x", encoding="utf-8") as stream:
        json.dump({"policy_sha256": report["policy_sha256"]}, stream)
    rows = evaluate(policy, cases)
    pending = result_file.with_suffix(".pending")
    save(pending, dict(policy_sha256=report["policy_sha256"], maps=rows))
    pending.replace(result_file)
    print(f"Test: {sum(r['outcome'] == 'SUCCESS' for r in rows)}/{len(rows)}. {result_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "watch", "test"))
    parser.add_argument("--run", type=Path, default=RESULTS / "run-02", help="папка отдельного обучения")
    args = parser.parse_args()
    try:
        if args.command == "train":
            return 0 if train(args.run) else 1
        if args.command == "watch":
            from .viewer import show
            values = read(args.run / "policy.json")["values"]
            levels = {c.name: c.level for c in load_split("val")}
            show(levels, None, lambda: Sarsa(values))
            return 0
        final_test(args.run)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
