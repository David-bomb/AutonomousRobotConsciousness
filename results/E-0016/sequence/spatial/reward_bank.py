"""Независимые старты для обучения по реальному финишу после локального прогноза."""
from hashlib import sha256
import json
from maze.datasets import geometry
from ...data import OUT, PACK, digest, generate, rows, save
from .fresh import HERE, fingerprints


def main():
    used = fingerprints()
    for path in (OUT/"data/train.jsonl", OUT/"data/dev.jsonl", OUT/"data/confirmation.jsonl",
                 HERE.parent/"train.jsonl", HERE.parent/"probe.jsonl", HERE/"extra.jsonl",
                 HERE/"selection.jsonl", HERE/"large.jsonl"):
        used.update(row["geometry"] for row in rows(path))
    excluded = sha256("\n".join(sorted(used)).encode()).hexdigest()
    groups = {}
    for name, seeds in (("screen", [9164100]), ("selection", [9164101, 9164102]),
                        ("probe", [9164103])):
        target = HERE/f"reward-{name}.jsonl"
        records = []
        for seed in seeds:
            records.extend(generate(("local-"+name, seed)))
        for row in records:
            key = geometry(row["grid"])
            assert key == row["geometry"] and key not in used
            used.add(key)
        with target.open("x", encoding="utf8", newline="\n") as f:
            for row in records:
                f.write(json.dumps(row, separators=(",", ":"))+"\n")
        groups[name] = dict(seeds=seeds, count=len(records), sha256=digest(target))
    save(HERE/"reward-bank.json", dict(excluded_sha256=excluded, groups=groups,
         generator=digest(PACK/"make_hard_maps.py")))


if __name__ == "__main__":
    main()
