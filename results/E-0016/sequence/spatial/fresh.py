"""Дополнительные геометрии. Из старых контрольных файлов извлекаются только отпечатки стен."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from tempfile import TemporaryDirectory
from pathlib import Path
import gzip
import json
import re
from maze.datasets import geometry,load_split
from ...data import OUT,ROOT,PACK,CORPUS,STRESS,copy_corpus,generate,read,rows,save,digest
HERE=Path(__file__).resolve().parent


def fingerprints():
    used=set(read(PACK/"excluded.json"))
    used.update(read(ROOT/"results/ARC-dot/replay/e6-fresh/diagnosis.json")["geometries"].values())
    used.update(geometry(c.level.grid) for split in ("train","val") for c in load_split(split))
    used.update(r["geometry"] for r in read(ROOT/"maze/datasets/test-manifest.json").values())
    field=re.compile(rb'"geometry"\s*:\s*"([0-9a-f]{64})"')
    for path in STRESS:
        with path.open("rb") as f:
            for line in f:used.add(field.search(line).group(1).decode())
    with TemporaryDirectory() as folder:
        db=copy_corpus(folder)
        used.update(h for h, in db.execute("SELECT hash FROM geoms"))
        db.close()
    for split in ("control","train"):
        with gzip.open(CORPUS.parent/f"diagnostics/{split}.jsonl.gz","rb") as stream:
            for line in stream:used.add(field.search(line).group(1).decode())
    archived=read(OUT/"data/manifest.json")
    assert len(used)==archived["excluded_count"]
    assert sha256("\n".join(sorted(used)).encode()).hexdigest()==archived["excluded_sha256"]
    return used


def main():
    used=fingerprints()
    for p in (OUT/"data/train.jsonl",OUT/"data/dev.jsonl",OUT/"data/confirmation.jsonl",
              HERE.parent/"train.jsonl",HERE.parent/"probe.jsonl"):
        used.update(r["geometry"] for r in rows(p))
    reg=dict(excluded=len(used),generator=digest(PACK/"make_hard_maps.py"),sets={})
    with ProcessPoolExecutor(2) as pool:
        for name,seeds in (("extra",range(9162100,9162132)),("selection",range(9162200,9162202))):
            path=HERE/(name+".jsonl")
            if path.exists():
                bank=rows(path)
            else:
                bank=[]
                for part in pool.map(generate,[("spatial-"+name,s) for s in seeds]):
                    bank.extend(part)
                    if len(bank)%400==0:print(name,len(bank),flush=True)
            for r in bank:
                key=geometry(r["grid"])
                assert r["geometry"]==key and key not in used
                used.add(key)
            assert len(bank)==len(seeds)*100
            if not path.exists():
                with path.open("x",encoding="utf8",newline="\n") as f:
                    for r in bank:f.write(json.dumps(r,separators=(",",":"))+"\n")
            reg["sets"][name]=dict(n=len(bank),seeds=list(seeds),sha256=digest(path))
    save(HERE/"fresh.json",reg)
    print(reg,flush=True)


if __name__=="__main__":
    main()
