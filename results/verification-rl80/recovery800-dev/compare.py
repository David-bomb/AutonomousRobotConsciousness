"""Сверка независимой поездки с опубликованной физической записью recovery800 по ID карты."""
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import checked_lock, receipt
from suite import ROOT, check, checked_suite, file_hash, read, write_new

LOCK_SHA = "c39f348a742f579bc28d23bf5b126739c29bef6f29b9e8a29a577f20a0d5fc38"
COLUMNS = ("outcome", "steps", "refuels", "repairs")


def by_id(rows, key):
    result = {r[key]: r for r in rows}
    check(len(rows) == len(result) == 100, "Нужно ровно по одной поездке на каждой карте dev")
    return result


def totals(rows):
    return dict(outcomes=dict(Counter(r["outcome"] for r in rows)),
                steps=sum(r["steps"] for r in rows), refuels=sum(r["refuels"] for r in rows),
                repairs=sum(r["repairs"] for r in rows),
                both_services=sum(r["refuels"] > 0 and r["repairs"] > 0 for r in rows))


def main():
    provenance = read(HERE / "provenance.json")
    lock = checked_lock(HERE / "lock.json", LOCK_SHA)
    cases = checked_suite(ROOT / lock["spec"])
    independent = HERE / "independent"
    verified = receipt(cases, independent, lock, LOCK_SHA)
    check(verified == read(independent / "summary.json"), "Итог оценщика расходится с журналом")
    path = HERE / "rl.physical.json"
    check(file_hash(path) == provenance["reference_sha256"], "Изменился опорный журнал")
    baseline = read(path)
    check(baseline["weights_sha256"] == lock["checkpoint_sha256"] == provenance["checkpoint_sha256"],
          "Сравниваются разные веса")
    check(baseline["dev_sha256"] == file_hash(ROOT / "results/E-0016/data/dev.jsonl") == provenance["dev_sha256"],
          "Сравниваются разные наборы dev")
    for source in ("adapter.py", "policy.py", "geometry.cpp", "geometry.dll"):
        check(file_hash(HERE / "frozen" / source) == baseline["sources"][source],
              "Реализация политики расходится с регистрацией исходных поездок")
    old = by_id(baseline["episodes"], "map")
    new = by_id([json.loads(line) for line in (independent / "episodes.jsonl").read_bytes().splitlines()], "id")
    check(old.keys() == new.keys() == {c.name for c in cases}, "Различается состав карт")
    diffs, equal = [], dict.fromkeys(COLUMNS, 0)
    for name in old:
        a, b = old[name], new[name]
        different = []
        for k in COLUMNS:
            if type(a[k]) is type(b[k]) and a[k] == b[k]:
                equal[k] += 1
            else:
                different.append(k)
        if different:
            diffs.append(dict(id=name, fields=different, reference={k: a[k] for k in COLUMNS},
                              independent={k: b[k] for k in COLUMNS}))
    comparison = dict(**verified, reference_sha256=file_hash(path), reference=totals(old.values()),
                      independent=totals(new.values()), matched_in_all_fields=100-len(diffs),
                      matched_per_field=equal, differences=diffs)
    target = HERE / "receipt.json"
    if target.exists():
        check(comparison == read(target), "Ранее сохранённая сверка изменилась")
    else:
        write_new(target, comparison)
    print(json.dumps({k: comparison[k] for k in ("matched_in_all_fields", "matched_per_field",
                                                 "independent", "reference")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
