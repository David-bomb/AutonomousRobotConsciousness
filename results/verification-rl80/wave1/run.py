"""Три независимых сотни. На каждой карте отдельный CPU-процесс и штатный шаг среды."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import checked_lock, episode, receipt
from suite import ROOT, SPLITS, check, checked_suite, digest, file_hash, inventory, read, write_new

OUT = HERE / "independent"


def play(cases, adapter, weights, weights_sha):
    with (OUT / (cases[0].split + ".jsonl")).open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            row = episode(case, adapter, weights)
            row["checkpoint_sha256"] = weights_sha
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()


def registered(lock_sha):
    meta = read(HERE / "registered.json")
    lock = checked_lock(HERE / "lock.json", lock_sha)
    selected = read(HERE / "selection.json")
    check(file_hash(HERE / "selection.json") == meta["selection_sha256"]
          and meta["lock_sha256"] == lock_sha and f'runs/{meta["candidate"]}.pt' == selected["weights"]
          and meta["weights_sha256"] == selected["weights_sha256"] == lock["checkpoint_sha256"],
          "Не совпадает регистрация первой волны")
    return meta, lock


def check_adapter(meta, adapter, weights):
    cases, _ = inventory([ROOT / "results/E-0016/data/dev.jsonl"], "validation")
    case = cases[0]
    dev_path = ROOT / "results/E-0016" / meta["original_weights"]
    reference = dev_path.with_suffix(".physical.json")
    check(file_hash(reference) == meta["dev_physical_journals_sha256"][meta["candidate"]],
          "Изменился журнал физической dev")
    saved = next(r for r in read(reference)["episodes"] if r["map"] == case.name)
    actual = episode(case, adapter, weights)
    check(all(saved[k] == actual[k] for k in ("outcome", "steps", "refuels", "repairs")),
          "Замороженная цепочка не воспроизвела исходную поездку dev")
    return dict(matched_fields=["outcome", "steps", "refuels", "repairs"],
                actions_sha256=actual["actions_sha256"], weights_sha256=meta["weights_sha256"])


def verify(meta, lock, lock_sha, cases):
    summary = receipt(cases, OUT, lock, lock_sha)
    by_split, blocks = {}, []
    for split in SPLITS:
        data = (OUT / (split + ".jsonl")).read_bytes()
        trips = [json.loads(line) for line in data.splitlines()]
        check(len(trips) == 100 and all(r["split"] == split for r in trips), "Неполный журнал сотни")
        outcomes = Counter(r["outcome"] for r in trips)
        by_split[split] = dict(maps=100, success=outcomes["SUCCESS"], outcomes=dict(outcomes),
                               steps=sum(r["steps"] for r in trips), refuels=sum(r["refuels"] for r in trips),
                               repairs=sum(r["repairs"] for r in trips), episodes_sha256=digest(data))
        blocks.append(data)
    check(b"".join(blocks) == (OUT / "episodes.jsonl").read_bytes(), "Покартовые журналы расходятся")
    full = dict(**summary, candidate=meta["candidate"], selection_sha256=meta["selection_sha256"],
                registered_utc=meta["registered_utc"], by_split=by_split,
                steps=sum(t["steps"] for t in by_split.values()),
                unique_ids=len({c.name for c in cases}), unique_geometries=len({c.geometry for c in cases}))
    return summary, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "verify"))
    p.add_argument("--lock-sha256", required=True)
    args = p.parse_args()
    meta, lock = registered(args.lock_sha256)
    if args.action == "run":
        OUT.mkdir(exist_ok=False)
        adapter = (ROOT / lock["adapter"]).read_bytes()
        weights = (ROOT / lock["checkpoint"]).read_bytes()
        check(digest(adapter) == lock["adapter_sha256"] and digest(weights) == meta["weights_sha256"],
              "Подмена переданных байтов")
        write_new(OUT / "adapter-check.json", check_adapter(meta, adapter, weights))
    cases = checked_suite(ROOT / lock["spec"])
    if args.action == "run":
        write_new(OUT / "start.json", dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                           lock_sha256=args.lock_sha256, cpu_processes=3))
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(play, [c for c in cases if c.split == s], adapter, weights,
                                 meta["weights_sha256"]) for s in SPLITS]
            for task in tasks:
                task.result()
        with (OUT / "episodes.jsonl").open("xb") as merged:
            for split in SPLITS:
                merged.write((OUT / (split + ".jsonl")).read_bytes())
        checked_lock(HERE / "lock.json", args.lock_sha256)
    summary, full = verify(meta, lock, args.lock_sha256, cases)
    if args.action == "run":
        write_new(OUT / "summary.json", summary)
        write_new(HERE / "receipt.json", full)
    else:
        check(summary == read(OUT / "summary.json") and full == read(HERE / "receipt.json"),
              "Итог отличается от зарегистрированного")
    print(json.dumps(full, ensure_ascii=False))


if __name__ == "__main__":
    main()
