"""Пересчёт архивных исходов без загрузки сетей или повторных поездок."""
import argparse
from collections import Counter
import csv
import gzip
import json
import sqlite3

from suite import (HERE, ROOT, SPLITS, archived, check, file_hash, geometry, Level, lf_hash,
                   load_split, object_hash, read, write_new)

RESULTS = ROOT / "results"


def counts(rows, expected, outcome="outcome"):
    check(len(rows) == len(expected) and {(r["split"], r["map"]) for r in rows} == expected,
          "Журнал содержит чужие, недостающие или повторные карты")
    wins = dict.fromkeys(sorted({s for s, _ in expected}), 0)
    for r in rows:
        check(r[outcome] in {"SUCCESS", "EMPTY", "BROKEN", "LIMIT"}, "Неизвестный исход")
        wins[r["split"]] += r[outcome] == "SUCCESS"
    return wins


def automata(expected):
    folder = RESULTS / "E-0010"
    coverage = read(folder / "coverage-177-f9836b733c39.json")
    expected = expected | {("val", c.name) for c in load_split("val")}
    series = {}
    for name, fingerprint in coverage["source_sha256_lf"].items():
        path = folder / (name + ".jsonl")
        check(lf_hash(path) == fingerprint, "Изменился исходный журнал автомата")
        series[name] = counts([json.loads(l) for l in path.read_bytes().splitlines()], expected, "status")
    check(len(series) == coverage["series"] == 177, "Неполный срез автоматов")
    return dict(series=len(series), episodes=len(series)*len(expected), reserve=series["reserve"],
                maxima={s: max(v[s] for v in series.values()) for s in SPLITS},
                best_total=max(sum(v[s] for s in SPLITS) for v in series.values()),
                full_val=sum(v["val"] == 36 for v in series.values()))


def e13(expected):
    folder = RESULTS / "E-0013"
    models = read(folder / "manifest.json")["models"]
    with (folder / "scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    check(len(models) == len(table) == 204 and {r["model"] for r in table} == set(models),
          "Неполный реестр E-0013")
    by_total, winners, hashes = Counter(), {}, {}
    both_services = 0
    for score in table:
        name = score["model"]
        model = models[name]
        path = folder / "models" / (name + ".json")
        log = read(path)
        hashes[name] = file_hash(path)
        check(log["model"] == name and log["sha256"] == model["sha256"] == file_hash(ROOT / model["weights"]),
              "Несогласованные веса E-0013")
        rows = log["rows"]
        wins = counts(rows, expected)
        check(wins == log["wins"], "Сводка E-0013 расходится с поездками")
        total = sum(wins.values())
        outcomes = Counter(r["outcome"] for r in rows)
        both = sum(r["refuels"] > 0 and r["repairs"] > 0 for r in rows)
        stats = dict(**wins, total=total, broken=outcomes["BROKEN"], empty=outcomes["EMPTY"],
                     limit=outcomes["LIMIT"], both_services=both,
                     serviced=sum(r["refuels"] + r["repairs"] > 0 for r in rows))
        check(all(int(score[k]) == n for k, n in stats.items()), "Ошибка в scores.csv")
        by_total[total] += 1
        both_services += both
        if total:
            winners[name] = wins
    return dict(models=len(table), episodes=len(table)*300, histogram=dict(sorted(by_total.items())),
                total_success=sum(k*v for k, v in by_total.items()), winners=winners,
                both_services=both_services, logs_sha256=object_hash(hashes))


def e15(expected):
    folder = RESULTS / "E-0015"
    reg, published = read(folder / "registration.json"), read(folder / "comparison.json")
    totals, end_states = {}, {}
    stages = {i: Counter() for i in (1, 2, 3)}
    sizes = Counter()
    for arm in reg["arms"]:
        total, outcomes = Counter(), Counter()
        for cfg in reg["configs"]:
            run = folder / "runs" / cfg["id"] / arm
            training, log = read(run / "training.json"), read(run / "evaluation.json")
            saved = published["evidence"][run.relative_to(folder).as_posix()]
            check(training["config"] == cfg and training["arm"] == arm, "Не совпал запуск")
            check(training["sha256"] == log["sha256"] == saved["weights"] == file_hash(run / "weights.pt")
                  and file_hash(run / "training.json") == saved["training"]
                  and file_hash(run / "evaluation.json") == saved["evaluation"], "Изменился журнал E-0015")
            rows = log["stress"]
            wins = counts(rows, expected)
            check(wins == published["seeds"][cfg["id"]][arm]["stress_wins"], "Расхождение в E-0015")
            total.update(wins)
            total["stress"] += sum(wins.values())
            total["visited_both"] += sum(r["refuels"] > 0 and r["repairs"] > 0 for r in rows)
            outcomes.update(r["outcome"] for r in rows)
            for key, group, count in (("validation", "validation", 240), ("fork_finishes", "fork_episodes", 50)):
                trips = log[group]
                check(len(trips) == len({r["map"] for r in trips}) == count, "Неполный набор поездок")
                total[key] += sum(r["outcome"] == "SUCCESS" for r in trips)
            forks = log["forks"]
            check(len(forks) == len({r["pair"] for r in forks}) == 25, "Неполный набор развилок")
            for row in forks:
                a, b = row["choices"]
                x, y = row["expected"]
                total.update(both=int(a == x and b == y), correct=int(a == x)+int(b == y),
                             changed=int(a != b))
            for row in training["history"]:
                total.update(steps=row["steps"], updates=row["updates"])
                stages[row["stage"]].update(row["outcomes"])
            sizes[arm] += len(rows)
        check(dict(total) == published["totals"][arm], "Архивная сводка E-0015 расходится с журналами")
        totals[arm], end_states[arm] = dict(total), dict(outcomes)
    return dict(stress_episodes=dict(sizes), totals=totals, stress_outcomes=end_states,
                training_stages={k: dict(success=v["SUCCESS"], completed=sum(v.values())) for k,v in stages.items()})


def overlap(cases):
    folder = RESULTS / "E-0012"
    shapes = {c.geometry for c in cases}
    database = folder / "corpus.sqlite"
    check(file_hash(database) == read(RESULTS / "E-0015/registration.json")["train_db_sha256"],
          "Контрольная сумма корпуса изменилась")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        train = {r[0] for r in db.execute("SELECT hash FROM geoms WHERE split='train'")}
    training_file = folder / "diagnostics/train.jsonl.gz"
    check(file_hash(training_file) == read(folder / "diagnostics/verified.json")["sha256"]["train"],
          "Изменился учебный банк развилок")
    with gzip.open(training_file, "rt", encoding="utf-8") as stream:
        diag = {geometry(Level.parse(json.loads(line)["source"]).grid) for line in stream}
    return dict(corpus_train_unique=len(train), corpus_train_common=len(train & shapes),
                diagnostic_train_unique=len(diag), diagnostic_train_common=len(diag & shapes))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    cases, _ = archived()
    expected = {(c.split, c.name) for c in cases}
    audit = dict(E0010=automata(expected), E0013=e13(expected), E0015=e15(expected), overlap=overlap(cases))
    out = HERE / "history.json"
    if args.write:
        write_new(out, audit)
    else:
        check(object_hash(audit) == object_hash(read(out)), "Итоги пересчёта изменились")
    print("OK", object_hash(audit))
