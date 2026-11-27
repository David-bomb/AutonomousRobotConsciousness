"""Новые полные поездки. Отпечаток строится по стенам во всех восьми ориентациях."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import import_module
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

from maze.datasets import geometry, load_split
from maze.levels import Level

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PACK = ROOT / "results/ARC-dot/source/ARC-stress-100"
CORPUS = ROOT / "results/E-0012/corpus.sqlite"
STRESS = (PACK/"maps.jsonl", ROOT/"results/E-0007/fresh.jsonl", ROOT/"results/E-0009/fresh.jsonl")
SEEDS = {"train": list(range(9160100, 9160108)), "dev": [9160200], "confirmation": [9160300, 9160301]}


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def copy_corpus(directory):
    copy = Path(directory) / CORPUS.name
    shutil.copyfile(CORPUS, copy)
    wal = CORPUS.with_suffix(".sqlite-wal")
    if wal.exists():
        shutil.copyfile(wal, copy.with_suffix(".sqlite-wal"))
    return sqlite3.connect(copy)


def excluded():
    used = set(read(PACK/"excluded.json"))
    used.update(read(ROOT/"results/ARC-dot/replay/e6-fresh/diagnosis.json")["geometries"].values())
    used.update(geometry(c.level.grid) for split in ("train", "val") for c in load_split(split))
    used.update(r["geometry"] for r in read(ROOT/"maze/datasets/test-manifest.json").values())
    for path in STRESS:
        for row in rows(path):
            assert geometry(row["grid"]) == row["geometry"]
            used.add(row["geometry"])
    with TemporaryDirectory() as folder:
        db = copy_corpus(folder)
        used.update(h for h, in db.execute("SELECT hash FROM geoms"))
        db.close()
    for split in ("control", "train"):
        with gzip.open(CORPUS.parent/f"diagnostics/{split}.jsonl.gz", "rt") as stream:
            used.update(json.loads(line)["geometry"] for line in stream)
    return used


def generate(job):
    split, seed = job
    generator = import_module("results.ARC-dot.source.ARC-stress-100.make_hard_maps")
    generator.SEED = seed
    batch = list(generator.generate())
    for item in batch:
        item["name"] = f"{split}-{seed}-{item['name']}"
    return batch


def level(row):
    return Level.parse("\n".join([row["name"], f"{row['fuel']} {row['health']}", *row["grid"]]))


def main():
    folder = OUT/"data"
    folder.mkdir(exist_ok=True)
    used = excluded()
    record = {"excluded_count": len(used), "excluded_sha256": sha256("\n".join(sorted(used)).encode()).hexdigest(),
              "generator_sha256": digest(PACK/"make_hard_maps.py"), "seeds": SEEDS, "splits": {}}
    with ProcessPoolExecutor(4) as workers:
        for split, seeds in SEEDS.items():
            path = folder/f"{split}.jsonl"
            if path.exists():
                batch = rows(path)
            else:
                batch = [r for pack in workers.map(generate, [(split, s) for s in seeds]) for r in pack]
            assert len(batch) == 100 * len(seeds)
            for item in batch:
                key = geometry(item["grid"])
                assert key == item["geometry"] and key not in used
                used.add(key)
            if not path.exists():
                with path.open("x", encoding="utf8", newline="\n") as stream:
                    for item in batch:
                        stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            record["splits"][split] = {"count": len(batch), "sha256": digest(path)}
            print(split, record["splits"][split], flush=True)
    path = folder/"manifest.json"
    if path.exists():
        assert read(path) == record
    else:
        save(path, record)


if __name__ == "__main__":
    main()
