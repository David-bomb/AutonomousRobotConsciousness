"""Банк, регистрация и файлы отдельной ветви."""
import hashlib
import json
from pathlib import Path
import random
from datetime import datetime,timezone
from maze.datasets import geometry
from ..data import level,rows,read,save,digest

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
SEQ=PARENT/"sequence"
DEV=PARENT/"data/dev.jsonl"
TRAIN=SEQ/"train.jsonl"
BASE=SEQ/"runs/mission600/rl.pt"
BASE_SHA="b04e0e0a29c345fd29db17a443de2882072c831d36ff18c792e5098f3a0bc133"
SEED=161127


def write(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    save(path,obj)


def register():
    assert digest(BASE)==BASE_SHA and digest(TRAIN)==read(SEQ/"bank.json")["splits"]["train"]["sha256"]
    dev=rows(DEV); train=rows(TRAIN)
    assert digest(DEV)==read(PARENT/"data/manifest.json")["splits"]["dev"]["sha256"]
    hashes=[geometry(r["grid"]) for r in [*train,*dev]]
    assert hashes==[r["geometry"] for r in [*train,*dev]] and len(set(hashes))==1700
    rng=random.Random(SEED)
    groups={}
    for k,(lo,hi,n) in dict(clean=(0,400,128),deviate=(400,800,128),
                             terminal=(800,1200,256),extra=(1200,1500,200),
                             probe=(1500,1600,48)).items():
        bands=[rng.sample([r["name"] for r in train[lo:hi] if r["tier"]==f"H{i}"],n//4) for i in range(1,5)]
        groups[k]=[s for line in zip(*bands,strict=True) for s in line]
    assert len(set.union(*(set(s) for s in groups.values())))==sum(map(len,groups.values()))
    root=PARENT.parents[1]
    files=[HERE/"protocol.md",HERE/"common.py",SEQ/"model.py",SEQ/"geometry.cpp",SEQ/"sequence.dll",
           PARENT/"policy.py",PARENT/"geometry.dll",PARENT/"critical_policy.py",PARENT/"rich_policy.py",
           *[root/"maze"/(n+".py") for n in ("interface","environment","grid","levels","memory")]]
    item=dict(base_sha256=BASE_SHA,train_sha256=digest(TRAIN),dev_sha256=digest(DEV),
              clean_trace_sha256=digest(SEQ/"clean-400.json"),groups=groups,overlap=0,
              sources={f.relative_to(root).as_posix():digest(f) for f in files})
    out=HERE/"registration.json"
    if out.exists():
        assert all(read(out)[k]==v for k,v in item.items())
    else:
        write(out,dict(**item,created=datetime.now(timezone.utc).isoformat()))


def group(kind):
    reg=read(HERE/"registration.json")
    assert digest(TRAIN)==reg["train_sha256"]
    by_name={r["name"]:r for r in rows(TRAIN)}
    return [by_name[name] for name in reg["groups"][kind]]
