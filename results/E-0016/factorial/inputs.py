"""Локальная копия разрешённых записей E-0012 и регистрация до обучения."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import zlib

import numpy as np
import torch

from maze.datasets import geometry
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SNAPSHOT = OUT / "train-dev.sqlite"
INDEX = OUT / "scenarios.json.gz"
REG = OUT / "registration.json"
SPLITS = {"train": 30000, "validation_id": 60}


def weights_hash(net):
    checksum = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        checksum.update(name.encode() + b"\0")
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def snapshot():
    archive = ROOT / "results/E-0012/corpus.sqlite"
    # SQLite может менять shm даже при mode=ro. WAL воспроизводим на частной копии.
    with TemporaryDirectory(dir=OUT) as tmp:
        local = Path(tmp) / archive.name
        sources = {}
        for suffix in ("", "-wal"):
            original = Path(str(archive) + suffix)
            if original.exists():
                copied = Path(str(local) + suffix)
                shutil.copyfile(original, copied)
                sources[original.name] = digest(copied)
                assert sources[original.name] == digest(original)
        dest = Path(tmp) / "selected.sqlite"
        with sqlite3.connect(dest) as db:
            db.execute("ATTACH DATABASE ? AS archive", (str(local),))
            db.execute("CREATE TABLE geoms (split TEXT, idx INTEGER, hash TEXT UNIQUE, size INTEGER, "
                       "data BLOB, PRIMARY KEY(split,idx))")
            for split, count in SPLITS.items():
                db.execute("INSERT INTO geoms SELECT split,idx,hash,size,data FROM archive.geoms "
                           "WHERE split=? AND idx>=0 AND idx<? ORDER BY idx", (split,count))
        db.close()
        dest.replace(SNAPSHOT)
    return sources


def index_cases():
    index, seen = {}, set()
    summary = {}
    translate = str.maketrans("ABFR", "....")
    db = sqlite3.connect(SNAPSHOT.as_uri() + "?immutable=1", uri=True)
    for split, count in SPLITS.items():
        hashes = []
        for idx, key, size, data in db.execute(
            "SELECT idx,hash,size,data FROM geoms WHERE split=? ORDER BY idx", (split,)
        ):
            assert idx == len(hashes) and key not in seen
            hashes.append(key)
            seen.add(key)
            pack = json.loads(zlib.decompress(data))
            assert len(pack) == 4
            walls = None
            for j, record in enumerate(pack):
                source = record["source"]
                name, stocks, grid = source.split("\n", 2)
                assert name == f"{split}-{idx:05}-{j}" and record["proof"]["status"] == "SUCCESS"
                assert hashlib.sha256(source.encode()).hexdigest() == record["sha256"]
                if walls is None:
                    walls = grid.translate(translate)
                    assert geometry(tuple(walls.splitlines())) == key
                else:
                    assert walls == grid.translate(translate)
                fuel, health = map(int, stocks.split())
                length = record["direct"]
                index[name] = dict(geometry=key, direct=length, fuel=fuel, health=health,
                                   deficits=int(fuel < length) + int(health < length),
                                   stage=1 if size <= 15 else 2 if size <= 21 else 3,
                                   source_sha256=record["sha256"])
        assert len(hashes) == count
        summary[split] = dict(geometries=count, scenarios=count * 4,
                              geometry_sha256=hashlib.sha256("\n".join(hashes).encode()).hexdigest())
    db.close()
    train = sorted((k for k in index if k.startswith("train-")),
                   key=lambda k: (index[k]["deficits"], index[k]["direct"], k))
    for i, name in enumerate(train):
        index[name]["c1"] = 1 + i // (len(train)//3)
    raw = json.dumps(index, separators=(",", ":"), sort_keys=True).encode()
    INDEX.write_bytes(gzip.compress(raw, compresslevel=3, mtime=0))
    return summary, hashlib.sha256(raw).hexdigest()


def register():
    if REG.exists():
        raise FileExistsError("Регистрация уже сохранена")
    base = read(ROOT / "results/E-0015/registration.json")
    assert all(digest(ROOT / p) == h for p, h in base["base_sources"].items())
    source_files = ["results/E-0015/registration.json", "results/E-0016/protocol.md",
                    "results/E-0016/reward_test.py", "results/E-0016/data.py"]
    source_files += [p.relative_to(ROOT).as_posix() for p in
                     (OUT / name for name in ("plan.md", "inputs.py", "run.py"))]
    sources = snapshot()
    assert sources["corpus.sqlite"] == base["train_db_sha256"]
    splits, content = index_cases()
    configs = base["configs"]
    initial = {}
    for cfg in configs:
        torch.manual_seed(cfg["seed"])
        initial[str(cfg["seed"])] = weights_hash(Network(cfg))
    save(REG, dict(created_utc=datetime.now(timezone.utc).isoformat(), configs=configs,
                   source_db=sources, selected_db_sha256=digest(SNAPSHOT), splits=splits,
                   train_dev_overlaps=0, index_sha256=content, index_gz_sha256=digest(INDEX),
                   initial=initial, base_sources=base["base_sources"],
                   sources={p:digest(ROOT / p) for p in source_files}, input_canvas=base["input_canvas"],
                   platform=platform.platform(), python=platform.python_version(),
                   torch=str(torch.__version__), numpy=np.__version__, device="cpu"))
    print("registered", splits, flush=True)
