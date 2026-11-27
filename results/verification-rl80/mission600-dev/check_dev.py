"""Полные поездки на dev и независимая сверка с первичным журналом по ID."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import checked_lock, episode, receipt
from suite import ROOT, check, checked_suite, digest, file_hash, read, write_new

OUT = HERE / "independent"
WEIGHTS_SHA = "b04e0e0a29c345fd29db17a443de2882072c831d36ff18c792e5098f3a0bc133"
FIELDS = ("outcome", "steps", "refuels", "repairs")


def compare(rows, summary, lock):
    reference = read(HERE / "rl.physical.json")
    check(reference["weights_sha256"] == lock["checkpoint_sha256"] == WEIGHTS_SHA, "Чужие веса")
    dev = ROOT / "results/E-0016/data/dev.jsonl"
    check(file_hash(dev) == reference["dev_sha256"], "Опорный журнал относится к другому dev")
    old = {r["map"]: r for r in reference["episodes"]}
    check(len(old) == len(reference["episodes"]) == len(rows) == 100 and old.keys() == {r["id"] for r in rows},
          "Различается состав поездок")
    differences, equal = [], dict.fromkeys(FIELDS, 0)
    for row in rows:
        prior = old[row["id"]]
        matches = []
        for k in FIELDS:
            match = type(row[k]) is type(prior[k]) and row[k] == prior[k]
            equal[k] += match
            matches.append(match)
        if not all(matches):
            differences.append(dict(id=row["id"], independent={k: row[k] for k in FIELDS},
                                    reference={k: prior[k] for k in FIELDS}))
    tier = {r["name"]: r["tier"] for r in (json.loads(x) for x in dev.read_bytes().splitlines())}
    wins = Counter(tier[r["id"]] for r in rows if r["outcome"] == "SUCCESS")
    return dict(**summary, reference_sha256=file_hash(HERE / "rl.physical.json"),
                matched_in_all_fields=100-len(differences), matched_per_field=equal,
                differences=differences, by_tier={t: wins[t] for t in sorted(set(tier.values()))},
                outcomes=dict(Counter(r["outcome"] for r in rows)),
                steps=sum(r["steps"] for r in rows), refuels=sum(r["refuels"] for r in rows),
                repairs=sum(r["repairs"] for r in rows),
                both_services=sum(r["repairs"] > 0 and r["refuels"] > 0 for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    lock = checked_lock(HERE / "lock.json", args.lock_sha256)
    check(lock["checkpoint_sha256"] == WEIGHTS_SHA and lock["spec_sha256"] ==
          "99a185a1425b1a236400ae2466bbbefc3410e1f468f85e368614ce6d2c5099ab", "Нужна регистрация mission600/dev")
    cases = checked_suite(ROOT / lock["spec"])
    if args.action == "run":
        OUT.mkdir(exist_ok=False)
        adapter = (ROOT / lock["adapter"]).read_bytes()
        weights = (ROOT / lock["checkpoint"]).read_bytes()
        check(digest(weights) == WEIGHTS_SHA and digest(adapter) == lock["adapter_sha256"], "Подмена снимка")

        def play(case):
            row = episode(case, adapter, weights)
            return dict(**row, checkpoint_sha256=WEIGHTS_SHA)

        with (OUT / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            with ThreadPoolExecutor(max_workers=3) as pool:
                for row in pool.map(play, cases):
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
        checked_lock(HERE / "lock.json", args.lock_sha256)
    summary = receipt(cases, OUT, lock, args.lock_sha256)
    rows = [json.loads(line) for line in (OUT / "episodes.jsonl").read_bytes().splitlines()]
    report = compare(rows, summary, lock)
    if args.action == "run":
        write_new(OUT / "summary.json", summary)
        write_new(HERE / "receipt.json", report)
    else:
        check(read(OUT / "summary.json") == summary and read(HERE / "receipt.json") == report,
              "Сводка отличается от сохранённой")
    print(json.dumps({k: report[k] for k in ("successes", "outcomes", "steps", "matched_in_all_fields", "by_tier")}))


if __name__ == "__main__":
    main()
