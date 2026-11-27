"""Регистрация одного снимка, изолированная поездка и сверка контрольных сумм действий."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import checked_lock, episode, lock_data, receipt
from suite import ROOT, check, checked_suite, file_hash, object_hash, read, write_new

WEIGHTS = "c921c6cfe26fee80b0ba5bd05946de47012a5d1da35d308092d2116350b2493b"
DEV_SPEC = HERE.parent/"dev-spec.json"
DEV_SHA = "4ce3fcd9106332497bf6a6e6ad26040aae303fc297a77dc7606828c5edabe8a6"
FIELDS = ("outcome", "steps", "refuels", "repairs", "actions_sha256")


def register():
    origins = read(HERE/"origin.json")
    for path, sha in origins["file_sha256"].items():
        check(file_hash(HERE/path) == sha, f"Изменился файл снимка {path}")
    check(all(origins["runtime_dependency_matches_mission600_lock"].values()), "Дрейф зависимостей")
    snapshot = HERE/"frozen/model.pt"
    deps = [HERE/p for p in origins["file_sha256"] if HERE/p != snapshot]
    lock = lock_data(HERE/"bridge.py",snapshot,WEIGHTS,DEV_SPEC,deps+[HERE/"origin.json",Path(__file__)])
    checked_suite(DEV_SPEC)
    write_new(HERE/"lock.json",lock)
    print(object_hash(lock))


def compare(rows, summary, lock):
    reference = read(HERE/"packed-dev.json")
    check(reference["model_sha256"] == lock["checkpoint_sha256"] == WEIGHTS and
          reference["adapter_sha256"] == file_hash(HERE/"frozen/frontier_adapter.py") and
          reference["decision_sha256"] == file_hash(HERE/"decision.json"), "Не совпадает происхождение")
    dev = ROOT/"results/E-0016/data/dev.jsonl"
    check(file_hash(dev) == reference["dev_sha256"] == DEV_SHA, "Другой dev")
    cases = {r["name"]: r for r in map(json.loads, dev.read_bytes().splitlines())}
    old = {r["map"]:r for r in reference["episodes"]}
    check(len(old)==len(reference["episodes"])==len(rows)==100 and
          old.keys()==cases.keys()=={r["id"] for r in rows}, "Неполная или посторонняя сотня")
    matched = dict.fromkeys(FIELDS,0)
    tiers = dict.fromkeys(("H1","H2","H3","H4"),0)
    differences = []
    for row in rows:
        name = row["id"]
        prior, case = old[name], cases[name]
        check(prior["geometry"] == case["geometry"] == row["geometry"] and
              prior["tier"] == case["tier"], "Изменились геометрия или класс")
        tiers[case["tier"]] += row["outcome"] == "SUCCESS"
        unequal = []
        for key in FIELDS:
            same = type(row[key]) is type(prior[key]) and row[key] == prior[key]
            matched[key] += same
            if not same:
                unequal.append(key)
        if unequal:
            differences.append(dict(id=name,fields=unequal,reference={k:prior[k] for k in FIELDS},
                                    independent={k:row[k] for k in FIELDS}))
    reference_tiers = Counter(r["tier"] for r in old.values() if r["outcome"] == "SUCCESS")
    check(reference_tiers == reference["tiers"] and Counter(r["outcome"] for r in old.values()) == reference["outcomes"]
          and sum(r["steps"] for r in old.values()) == reference["steps"], "Итог исходного журнала не сходится")
    return dict(**summary, source_receipt_sha256=file_hash(HERE/"packed-dev.json"),
                original_adapter_sha256=reference["adapter_sha256"], action_encoding="SHA-256(raw uint8 0..7): N E S W NE SE SW NW",
                matched_all=100-len(differences),matched_per_field=matched,differences=differences,
                by_tier=tiers,outcomes=dict(Counter(r["outcome"] for r in rows)),
                steps=sum(r["steps"] for r in rows), refuels=sum(r["refuels"] for r in rows),
                repairs=sum(r["repairs"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register","run","verify"))
    parser.add_argument("--lock-sha256")
    parser.add_argument("--out", type=Path, default=HERE/"independent")
    args = parser.parse_args()
    if args.action == "register":
        register()
        return
    lock = checked_lock(HERE/"lock.json",args.lock_sha256)
    check(lock["checkpoint_sha256"] == WEIGHTS and ROOT/lock["spec"] == DEV_SPEC, "Нужен frontier48/dev")
    cases = checked_suite(DEV_SPEC)
    out = args.out
    if args.action == "run":
        out.mkdir(exist_ok=False)
        adapter = (ROOT/lock["adapter"]).read_bytes()
        weights = (ROOT/lock["checkpoint"]).read_bytes()

        def play(case):
            return dict(**episode(case,adapter,weights),checkpoint_sha256=WEIGHTS)

        with (out/"episodes.jsonl").open("x",encoding="utf-8",newline="\n") as f, ThreadPoolExecutor(3) as pool:
            for row in pool.map(play,cases):
                f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                f.flush()
        checked_lock(HERE/"lock.json",args.lock_sha256)
    summary = receipt(cases,out,lock,args.lock_sha256)
    rows = [json.loads(line) for line in (out/"episodes.jsonl").read_bytes().splitlines()]
    report = compare(rows,summary,lock)
    if args.action == "run":
        write_new(out/"summary.json",summary)
        write_new(out/"receipt.json",report)
    else:
        check(read(out/"summary.json")==summary and read(out/"receipt.json")==report,"Журналы или сводка изменились")
    print(json.dumps({k:report[k] for k in ("matched_all","matched_per_field","by_tier","steps","outcomes")}))


if __name__ == "__main__":
    main()
