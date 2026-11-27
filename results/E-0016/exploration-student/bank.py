"""Разрешённые данные пилота. Учебные и проверочные карты разделены по стенам."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from maze.datasets import geometry
from maze.levels import Level

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
TRAIN = PARENT/"sequence/train.jsonl"
DEV = PARENT/"data/dev.jsonl"
SEED = 160027


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
    pending.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def level(row):
    return Level.parse("\n".join([row["name"], f"{row['fuel']} {row['health']}", *row["grid"]]))


def train_order():
    registry = read(HERE/"registration.json")
    assert digest(TRAIN) == registry["train_sha256"]
    pool = {r["name"]:r for r in rows(TRAIN)}
    return [pool[n] for n in registry["ordered_train"]]


def register():
    assert digest(TRAIN) == read(PARENT/"sequence/bank.json")["splits"]["train"]["sha256"]
    assert digest(DEV) == read(PARENT/"data/manifest.json")["splits"]["dev"]["sha256"]
    candidates = [r for r in rows(TRAIN) if int(r["name"].split("-")[2]) in range(9161112,9161116)]
    evaluation = rows(DEV)
    hashes = [geometry(r["grid"]) for r in [*candidates,*evaluation]]
    assert hashes == [r["geometry"] for r in [*candidates,*evaluation]]
    assert len(candidates)==400 and len(evaluation)==100 and len(set(hashes))==500
    rng = random.Random(SEED)
    groups = [[r for r in candidates if r["tier"]==f"H{i}"] for i in range(1,5)]
    for group in groups:
        assert len(group)==100
        rng.shuffle(group)
    order = [r["name"] for tuple4 in zip(*groups,strict=True) for r in tuple4]
    sources = [HERE/"protocol.md", HERE/"bank.py", HERE/"collect.py"]
    sources += [ROOT/"maze"/(n+".py") for n in ("grid","interface","environment","memory","levels",
                "frontier_reserve","frontier_automaton","resource_automaton","pilot_automaton")]
    params = dict(train_sha256=digest(TRAIN),dev_sha256=digest(DEV),ordered_train=order,
                  dev_names=[r["name"] for r in evaluation],geometry_overlaps=0,
                  sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    path = HERE/"registration.json"
    if path.exists():
        prior = read(path)
        assert all(prior[k]==v for k,v in params.items())
    else:
        save(path,dict(**params,created=datetime.now(timezone.utc).isoformat()))
