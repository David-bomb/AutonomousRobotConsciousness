"""Сводка E-0011 и независимый повтор поездок лучших моделей из сохранённых весов."""
import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/E-0011"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(row):
    return (row["worst"], row["wins"], -(row["mean_success_steps"] or float("inf")),
            -row["variant"], -row["seed"])


def initial_policies(manifest, output):
    import torch
    from maze.datasets import load_split
    from neural_rl.experiment import registration, save
    from neural_rl.networks import Network
    from neural_rl.rollout import MapPool, evaluate
    if registration() != manifest:
        raise ValueError("Начальные веса нужно воссоздать в зарегистрированном окружении")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    pool = MapPool(load_split("val"))
    for config in manifest["runs"]:
        path = output/"initial"/config["id"]/"val.json"
        if path.exists():
            continue
        torch.manual_seed(config["seed"])
        net = Network(config).cuda()
        rows = evaluate(net, pool, config["horizon"])
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, dict(config=config, validation=rows))


def verify_policy(config, expected, pool, output):
    import torch
    from maze.environment import Status
    from maze.interface import Move
    from neural_rl.networks import Network, batch
    from neural_rl.perception import command, encode
    net = Network(config).cuda().eval()
    net.load_state_dict(torch.load(output/config["id"]/"weights.pt", weights_only=True))
    hierarchical = config["family"] == "hrl_astar"
    labels = ("finish", "fuel", "repair", "frontier") if hierarchical else tuple(m.name for m in Move)
    decisions = Counter()
    active = [(case, pool.start(case)) for case in pool.cases]
    observed = {}
    with torch.inference_mode():
        while active:
            states = [encode(env.view, hierarchical) for _, env in active]
            actions = net(batch(states))[0].argmax(1).tolist()
            remaining = []
            for (case, env), action in zip(active, actions):
                decisions[labels[action]] += 1
                for move in command(env.view, action, hierarchical, config["horizon"]):
                    env.step(move)
                    if env.status != Status.RUNNING:
                        break
                if env.status == Status.RUNNING:
                    remaining.append((case, env))
                else:
                    observed[case.name] = dict(map=case.name, stage=case.stage, outcome=env.status.name,
                                               steps=env.steps, refuels=env.refuels, repairs=env.repairs)
            active = remaining
    if observed != {r["map"]: r for r in expected}:
        raise ValueError(f"Расхождение с покартовой оценкой {config['id']}")
    return dict(id=config["id"], checked=len(observed), steps=sum(r["steps"] for r in observed.values()),
                decisions=dict(decisions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--initial", action="store_true", help="оценить модели до первого обновления весов")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    manifest = read(output/"manifest.json")
    if args.initial:
        initial_policies(manifest, output)
        return
    from maze.datasets import load_split
    expected_maps = {c.name: c.stage for c in load_split("val")}
    configs = {c["id"]: c for c in manifest["runs"]}
    rows, maps, grouped, reports = [], [], {}, {}
    for run_id, config in configs.items():
        folder = output/run_id
        report = read(folder/"run.json")
        history, validation = report["train"], report["validation"]
        if (report["config"] != config or sha256((folder/"weights.pt").read_bytes()).hexdigest()
                != report["weights_sha256"] or len(validation) != 36 or
                {r["map"]: r["stage"] for r in validation} != expected_maps or
                [s["stage"] for s in history] != [1, 2, 3] or
                any(s["steps"] != config["lanes"]*config["ticks"] or s["updates"] <= 0 for s in history)):
            raise ValueError(f"Неполная запись {run_id}")
        finished = [r for r in validation if r["outcome"] == "SUCCESS"]
        initial = read(output/"initial"/run_id/"val.json")
        if (initial["config"] != config or len(initial["validation"]) != 36
                or {r["map"]: r["stage"] for r in initial["validation"]} != expected_maps):
            raise ValueError(f"Неполная оценка начальных весов {run_id}")
        initial_wins = sum(r["outcome"] == "SUCCESS" for r in initial["validation"])
        before = {r["map"]: r for r in initial["validation"]}
        for trial in validation:
            base = before[trial["map"]]
            maps.append(dict(id=run_id, **trial, initial_outcome=base["outcome"], initial_steps=base["steps"]))
        stages = [sum(r["stage"] == i for r in finished) for i in (1, 2, 3)]
        outcomes = Counter(r["outcome"] for r in validation)
        if (min(stages), len(finished), dict(outcomes)) != (report["worst"], report["wins"], report["outcomes"]):
            raise ValueError(f"Не сходятся итоги {run_id}")
        row = dict(id=run_id, family=config["family"], variant=config["variant"], seed=config["seed"],
                   channels=config["architecture"][0], hidden=config["architecture"][1], depth=config["architecture"][2],
                   lr=config["lr"], easy=stages[0], medium=stages[1], hard=stages[2], worst=min(stages),
                   wins=len(finished), initial_wins=initial_wins, gain=len(finished)-initial_wins,
                   mean_success_steps=report["mean_success_steps"],
                   empty=outcomes["EMPTY"], broken=outcomes["BROKEN"], limit=outcomes["LIMIT"],
                   train_steps=sum(s["steps"] for s in history), updates=sum(s["updates"] for s in history),
                   decisions=sum(s["decisions"] for s in history), seconds=report["seconds"])
        rows.append(row)
        reports[run_id] = report
        grouped.setdefault(row["family"], []).append(row)
    if len(grouped) != 5 or any(len(group) != 50 for group in grouped.values()):
        raise ValueError("Для сводки нужны все 250 обучений")
    variants, families = [], {}
    for name, group in grouped.items():
        by_variant = []
        for variant in range(1, 11):
            seeds = [r for r in group if r["variant"] == variant]
            if len({r["seed"] for r in seeds}) != 5:
                raise ValueError("Нужны пять зёрен на вариант")
            line = dict(family=name, variant=variant, mean_wins=mean(r["wins"] for r in seeds),
                        mean_worst=mean(r["worst"] for r in seeds), min_wins=min(r["wins"] for r in seeds),
                        max_wins=max(r["wins"] for r in seeds),
                        mean_easy=mean(r["easy"] for r in seeds), mean_medium=mean(r["medium"] for r in seeds),
                        mean_hard=mean(r["hard"] for r in seeds))
            by_variant.append(line)
        variants.extend(by_variant)
        best = max(group, key=rank)
        families[name] = dict(representative=best, median_wins=median(r["wins"] for r in group),
                              min_wins=min(r["wins"] for r in group), max_wins=max(r["wins"] for r in group),
                              best_variant=max(by_variant, key=lambda v: (v["mean_worst"], v["mean_wins"], -v["variant"])),
                              mean_initial=mean(r["initial_wins"] for r in group),
                              mean_final=mean(r["wins"] for r in group),
                              mean_gain=mean(r["gain"] for r in group),
                              improved=sum(r["gain"] > 0 for r in group),
                              tied=sum(r["gain"] == 0 for r in group),
                              declined=sum(r["gain"] < 0 for r in group),
                              failures={k: sum(r[k] for r in group) for k in ("empty", "broken", "limit")})
    write_csv(output/"runs.csv", rows)
    write_csv(output/"variants.csv", variants)
    write_csv(output/"maps.csv", maps)
    summary = dict(runs=len(rows), train_steps=sum(r["train_steps"] for r in rows),
                   updates=sum(r["updates"] for r in rows), val_attempts=36*len(rows), families=families)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if args.verify:
        import torch
        from neural_rl.experiment import registration
        from neural_rl.rollout import MapPool
        if registration() != manifest:
            raise ValueError("Код, выборки или окружение изменились после регистрации")
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        pool = MapPool(load_split("val"))
        audits = []
        for result in families.values():
            run_id = result["representative"]["id"]
            audits.append(verify_policy(configs[run_id], reports[run_id]["validation"], pool, output))
        (output/"verification.json").write_text(json.dumps(audits, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
