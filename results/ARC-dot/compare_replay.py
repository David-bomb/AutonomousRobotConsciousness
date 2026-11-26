"""Сопоставить готовые журналы независимого повтора с исходными измерениями dot."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
REPLAY = ROOT / "replay"


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(records, fields):
    result = {tuple(row[k] for k in fields): row for row in records}
    if len(result) != len(records):
        raise ValueError(f"Повторяется ключ {fields}")
    return result


def compare(actual, expected, keys, extra=False):
    a, b = indexed(actual, keys), indexed(expected, keys)
    differences = [k for k in b if a.get(k) != b[k]]
    if not extra:
        differences.extend(a.keys()-b.keys())
    return dict(compared=len(b), differences=sorted(differences), additional=len(a.keys()-b.keys()))


def pair(records, a, b):
    result = indexed(records, ("case", "agent"))
    common, wins, losses = [], 0, 0
    for name in {r["case"] for r in records}:
        x, y = (result[name, agent] for agent in (a, b))
        x_ok, y_ok = (r["status"] == "SUCCESS" for r in (x, y))
        wins += x_ok and not y_ok
        losses += y_ok and not x_ok
        if x_ok and y_ok:
            common.append((int(x["steps"]), int(y["steps"])))
    return dict(agents=[a,b], wins=wins, losses=losses, common=len(common),
                medians=[median(lengths) for lengths in zip(*common)])


def main():
    report = {}
    e4 = SOURCE / "ARC-frontier-experiment/measurements"
    expected, failures = [], []
    actual = rows(REPLAY/"e4/episodes.csv")
    diag = read(REPLAY/"e4/diagnosis.json")
    for group in diag["splits"]:
        expected.extend(rows(e4/group/"episodes.csv"))
        old = read(e4/group/"diagnosis.json")
        assert old["splits"][group] == diag["splits"][group]
        assert old["policy_sha256"] == diag["policy_sha256"]
        failures.extend(old["failures"])
    report["E4_episodes"] = compare(actual, expected, ("split", "case", "agent"))
    report["E4_diagnosis"] = compare(diag["failures"], failures, ("split","case","agent"))
    holdout = [r for r in actual if r["split"] == "holdout"]
    report["E4_pair"] = pair(holdout, "frontier", "resources")

    e5 = SOURCE/"ARC-frontier-reserve/measurements"
    new = rows(REPLAY/"e5-fresh/episodes.csv")
    old = read(e5/"diagnosis.json")
    diag = read(REPLAY/"e5-fresh/diagnosis.json")
    assert old["geometries"] == diag["geometries"] and old["source_sha256"] == diag["source_sha256"]
    report["E5_episodes"] = compare(new, rows(e5/"episodes.csv"), ("case","agent"))
    report["E5_diagnosis"] = compare(diag["failures"], old["failures"], ("case","agent"))
    report["E5_pair"] = pair(new,"frontier-reserve","frontier")
    dev = rows(REPLAY/"e5-development.csv")
    expected = [r for r in rows(e5/"development-summary.csv") if r["variant"] == "2111"]
    groups = {r["split"] for r in dev}
    assert len(indexed(dev, ("split","case"))) == 336 and len(groups) == len(expected) == 5
    for r in expected:
        tested = [t for t in dev if t["split"] == r["split"]]
        counts = Counter(t["status"] for t in tested)
        assert sum(int(t["steps"]) for t in tested) == int(r["total_steps"])
        assert all(counts[k.upper()] == int(r[k]) for k in ("success","empty","broken","limit"))
    report["E5_development"] = {g: sum(r["status"] == "SUCCESS" for r in dev if r["split"] == g)
                                for g in sorted(groups)}

    e6 = SOURCE/"ARC-frontier-auto/measurements"
    expected, actual = [], []
    for group in ("train","val","test","holdout","reserve-fresh"):
        archived = "archive" if group == "test" else group
        actual.extend(dict(split=group, **r) for r in rows(REPLAY/f"e6-{group}.csv"))
        for agent in "ABCDEF":
            for case, status, steps in read(e6/f"development/arc-i-{archived}-{agent}.json"):
                expected.append(dict(split=group, case=case, agent=agent, status=status, steps=str(steps)))
    report["E6_development"] = compare(actual, expected, ("split","case","agent"))
    report["E6_AB"] = {agent: [sum(r["status"] == "SUCCESS" for r in actual
                                   if r["agent"] == agent and r["split"] in subset)
                               for subset in (("train","val","test"),("holdout",),("reserve-fresh",))]
                       for agent in "ABCDEF"}
    diag = read(REPLAY/"e6-fresh/diagnosis.json")
    old_fails = []
    for agent in ("C","reserve","frontier"):
        new = rows(REPLAY/f"e6-fresh/{agent}.csv")
        report["E6_"+agent] = compare(new, rows(e6/f"fresh/{agent}.csv"), ("case","agent"))
        old = read(e6/f"fresh/{agent}-diagnosis.json")
        assert diag["geometries"] == old["geometries"]
        old_fails.extend(old["failures"])
    report["E6_diagnosis"] = compare(diag["failures"], old_fails, ("case","agent"))

    new = rows(REPLAY/"stress/episodes.csv")
    old = rows(SOURCE/"ARC-stress-100/episodes.csv")
    assert len(indexed(new, ("map","variant"))) == 1600
    assert len(indexed(old, ("map","variant"))) == 1400
    assert set(Counter(r["variant"] for r in new).values()) == {100}
    report["stress_all_fields"] = compare(new,old,("map","variant"),extra=True)
    def physical(data):
        return [{k:v for k,v in row.items() if k != "skipped_cycle_steps"} for row in data]
    report["stress_physical"] = compare(physical(new),physical(old),("map","variant"),extra=True)
    report["stress_by_tier"] = {agent: [sum(r["status"]=="SUCCESS" for r in new if
                                           r["variant"]==agent and r["tier"]==tier)
                                       for tier in ("H1","H2","H3","H4")]
                                for agent in sorted({r["variant"] for r in new})}
    report["stress_solved"] = len({r["map"] for r in new if r["status"]=="SUCCESS"})
    report["stress_old_solved"] = len({r["map"] for r in old if r["status"]=="SUCCESS"})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
