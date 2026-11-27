"""Одноразовый изолированный процесс. Из защищённых банков выходят только агрегаты отпечатков."""
from collections import Counter
import gzip
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import re
import sqlite3
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
E16=ROOT/"results/E-0016"
LAB=Path("C:/Users/David/pers_projects/Codex_research_lab/projects/diplom-rl80")
sys.path.insert(0,str(ROOT))
from maze.datasets import geometry

SEEDS=(9167001,9167002)
FIELD=re.compile(rb'"geometry"\s*:\s*"([a-f0-9]{64})"')


def sha(data):
    return sha256(data).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(path,obj):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)
        f.write("\n")


def keys(p):
    data=p.read_bytes()
    if p.suffix==".gz":
        data=gzip.decompress(data)
    return {m.decode() for m in FIELD.findall(data)}


def main():
    archive=import_module("results.E-0016.sequence.spatial.fresh")
    # Официальный проверенный объединённый реестр 79 186 прежних геометрий.
    blocked=archive.fingerprints()
    hist=read(HERE.parent/"stress-spec.json")["contract"]["sets"]
    protected={}
    for group,entry in hist.items():
        p=ROOT/entry["path"]
        assert sha(p.read_bytes().replace(b"\r\n",b"\n"))==entry["sha256_lf"]
        individual=keys(p)
        assert len(individual)==100 and individual<=blocked
        protected[group]=dict(count=len(individual),sha256_lf=entry["sha256_lf"])
    reg=read(E16/"sequence/spatial/bank-audit.json")
    for entry in reg["banks"]:
        p=E16/entry["path"]
        assert sha(p.read_bytes())==entry["sha256"]
        values=keys(p)
        assert len(values)==entry["maps"] and blocked.isdisjoint(values)
        blocked.update(values)
        if entry["path"] in ("data/dev.jsonl","data/confirmation.jsonl"):
            protected[entry["path"]]=dict(count=len(values),sha256=entry["sha256"])
    assert len(blocked)==reg["combined_count"]
    con=sqlite3.connect((E16/"factorial/train-dev.sqlite").resolve().as_uri()+"?mode=ro",uri=True)
    try:
        factorial={v for v, in con.execute("select hash from geoms")}
    finally:
        con.close()
    blocked.update(factorial)
    extras={}
    # Покрывает и альтернативные банки prev-two в JSON-массивах, и новые банки spatial.
    for p in sorted(E16.rglob("*")):
        if not p.is_file() or p.suffix not in (".json",".jsonl",".gz") or p.name.startswith("."):
            continue
        values=keys(p)
        if values-blocked:
            extras[p.relative_to(ROOT).as_posix()]=dict(fingerprints=len(values),additional=len(values-blocked))
            blocked.update(values)
    for p in [HERE.parent/"phase100-train/train.jsonl",LAB/"handover-search/train.jsonl",LAB/"return-search/bank.jsonl",LAB/"panel-search/holdout.jsonl"]:
        values=keys(p)
        assert values
        extras[p.name+"@"+p.parent.name]=dict(fingerprints=len(values),additional=len(values-blocked))
        blocked.update(values)
    assert all(re.fullmatch(r"[0-9a-f]{64}",x) for x in blocked)
    old_count=len(blocked)
    generator=import_module("results.ARC-dot.source.ARC-stress-100.make_hard_maps")
    groups={}
    new=set()
    for seed in SEEDS:
        generator.SEED=seed
        rows=list(generator.generate())
        for r in rows:
            r["name"]=f"wave3-select-{seed}-"+r["name"]
            assert r["geometry"]==geometry(r["grid"])
        shapes={r["geometry"] for r in rows}
        assert len(shapes)==len(rows)==100 and not (shapes&blocked or shapes&new)
        assert Counter(r["tier"] for r in rows)==dict.fromkeys(("H1","H2","H3","H4"),25)
        new.update(shapes)
        target=HERE/f"{seed}.jsonl"
        with target.open("x",encoding="utf-8",newline="\n") as f:
            for r in rows:f.write(json.dumps(r,separators=(",",":"))+"\n")
        groups[seed]=dict(sha256=sha(target.read_bytes()),canonical_walls_sha256=sha("\n".join(sorted(shapes)).encode()),
                          maps=len(rows),overlap=0)
    save(HERE/"bank-receipt.json",dict(seeds=SEEDS,sets=groups,mutual_overlap=0,protected=protected,
         prior_archive_and_registered=reg["combined_count"],factorial_distinct=len(factorial),extra_sources=extras,
         excluded_count=old_count,excluded_sha256=sha("\n".join(sorted(blocked)).encode()),
         generator_sha256=sha(Path(generator.__file__).read_bytes()),extractor_sha256=sha(Path(__file__).read_bytes())))
    print(old_count, groups)


if __name__=="__main__":
    main()
