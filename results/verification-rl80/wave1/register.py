"""Сверить выбор по пяти журналам dev и закрепить один снимок до первого контроля."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import lock_data
from suite import ARCHIVE, ROOT, check, digest, file_hash, object_hash, read, write_new

SOURCE = ROOT / "results/E-0016"
ORDER = ("pilot200/rl", "recovery800/rl", "critical800/rl", "rich800/rl1", "rich800/rl2")
MODULES = ("frozen_adapter.py", "adapter.py", "policy.py", "critical_policy.py",
           "rich_policy.py", "geometry.cpp", "geometry.dll")
TIERS = ("H1", "H2", "H3", "H4")
DEV_SHA = "4ce3fcd9106332497bf6a6e6ad26040aae303fc297a77dc7606828c5edabe8a6"
SUITE_SHA = "58f493edd1adb2e0011a0fec3a5b095e47145881783e9cf6252c7a8d902c3168"


def primary_scores(dev_bytes):
    rows = [json.loads(line) for line in dev_bytes.splitlines()]
    tier = {r["name"]: r["tier"] for r in rows}
    check(digest(dev_bytes) == DEV_SHA and len(rows) == len(tier) == 100
          and Counter(tier.values()) == dict.fromkeys(TIERS, 25), "Набор dev изменился")
    candidates, evidence = [], {}
    for name in ORDER:
        weights = SOURCE / "runs" / (name + ".pt")
        path = weights.with_suffix(".physical.json")
        log = read(path)
        check(log["weights_sha256"] == file_hash(weights) and log["dev_sha256"] == DEV_SHA,
              "Журнал dev относится к другим весам или картам")
        trips = log["episodes"]
        check(len(trips) == 100 and {r["map"] for r in trips} == tier.keys()
              and all(r["tier"] == tier[r["map"]] for r in trips), "Неполный журнал dev")
        wins = Counter(r["tier"] for r in trips if r["outcome"] == "SUCCESS")
        by_tier = {t: wins[t] for t in TIERS}
        check(log["summary"]["wins"] == sum(wins.values()) and log["summary"]["by_tier"] == by_tier,
              "Счёт dev расходится с первичными записями")
        candidates.append(dict(name=name, sha256=log["weights_sha256"], wins=sum(wins.values()),
                               min_tier=min(by_tier.values())))
        evidence[name] = file_hash(path)
    return candidates, evidence


def main():
    manifest_path = SOURCE / "frozen.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    candidates, logs = primary_scores((SOURCE / "data/dev.jsonl").read_bytes())
    check(manifest["candidates"] == candidates and manifest["dev_sha256"] == DEV_SHA,
          "Протокол выбора расходится с пересчётом dev")
    best = max(candidates, key=lambda r: (r["wins"], r["min_tier"]))
    checkpoint = SOURCE / "runs" / (best["name"] + ".pt")
    selected = read(checkpoint.with_suffix(".physical.json"))
    check(manifest["weights"] == checkpoint.relative_to(SOURCE).as_posix()
          and manifest["weights_sha256"] == best["sha256"]
          and manifest["physical_dev"] == selected["summary"]
          and manifest["repeated_identical_episodes"] == 100
          and manifest["adapter"] == "frozen_adapter.py", "Выбран другой кандидат или адаптер")
    check(SUITE_SHA == object_hash(read(ARCHIVE)), "Изменилась спецификация трёх сотен")
    versions = manifest["source_sha256"]
    check(set(MODULES) | {"dev.py", "freeze.py"} == set(versions), "Нужна сверка состава исходников")
    contents = {f: (SOURCE / f).read_bytes() for f in versions}
    check(all(digest(data) == versions[f] for f, data in contents.items()),
          "Текущие исходники отличаются от первой заморозки")
    weights = checkpoint.read_bytes()
    check(digest(weights) == best["sha256"], "Изменились выбранные веса")
    check(not (HERE / "lock.json").exists(), "Первая волна уже зарегистрирована")
    frozen = HERE / "frozen"
    frozen.mkdir(exist_ok=False)
    for f, data in contents.items():
        (frozen / f).write_bytes(data)
    (frozen / "weights.pt").write_bytes(weights)
    (HERE / "selection.json").write_bytes(manifest_bytes)
    selected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    deps = [frozen / f for f in MODULES] + [HERE / "run.py", HERE / "selection.json"]
    lock = lock_data(HERE / "bridge.py", frozen / "weights.pt", best["sha256"], ARCHIVE, deps)
    write_new(HERE / "lock.json", lock)
    registered = dict(candidate=best["name"], original_weights=manifest["weights"],
                      weights_sha256=best["sha256"], lock_sha256=object_hash(lock),
                      selection_sha256=digest(manifest_bytes), registered_utc=selected_at,
                      dev_physical_journals_sha256=logs, dev_wins=best["wins"], dev_min_tier=best["min_tier"],
                      historical_spec_sha256=SUITE_SHA)
    write_new(HERE / "registered.json", registered)
    print(registered["candidate"], registered["weights_sha256"], registered["lock_sha256"])


if __name__ == "__main__":
    main()
