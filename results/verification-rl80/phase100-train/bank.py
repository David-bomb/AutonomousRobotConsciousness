"""Детерминированное построение только нового train; посторонние выборки не загружаются."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))
from maze.datasets import geometry

SEED = 9164107
TRAIN = {"results/E-0016/data/train.jsonl":"524c9b98b433779eb5cf51cbb654fe59c8a15ed78338deae6dd442785e2db0f5",
         "results/E-0016/sequence/train.jsonl":"6affb0aa46384b2f3485e630524bb0dc02abb5b5f9901adee4f786b0d1ada763"}


def main():
    generator = import_module("results.ARC-dot.source.ARC-stress-100.make_hard_maps")
    generator.SEED = SEED
    used = set()
    for name, expected in TRAIN.items():
        data=(ROOT/name).read_bytes()
        assert sha256(data).hexdigest()==expected
        for row in map(json.loads,data.splitlines()):
            key=geometry(row["grid"])
            assert key==row["geometry"] and key not in used
            used.add(key)
    rows=list(generator.generate(used))
    assert Counter(r["tier"] for r in rows)==dict.fromkeys(("H1","H2","H3","H4"),25)
    for row in rows:
        row["name"]=f"phase-train-{SEED}-"+row["name"]
        key=geometry(row["grid"])
        assert key==row["geometry"] and key not in used
        used.add(key)
    with (HERE/"train.jsonl").open("x",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(",",":"))+"\n")
    print("train",len(rows),sha256((HERE/"train.jsonl").read_bytes()).hexdigest())


if __name__=="__main__":
    main()
