"""Расширение покрытия коротких истинных маршрутов на разных независимых геометриях."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
from maze.datasets import geometry
from ...data import OUT,PACK,read,rows,save,digest,generate

HERE=Path(__file__).resolve().parent


def main():
    from .fresh import fingerprints
    used=fingerprints()
    for p in (OUT/"data/train.jsonl",OUT/"data/dev.jsonl",OUT/"data/confirmation.jsonl",HERE.parent/"train.jsonl",
              HERE.parent/"probe.jsonl",HERE/"extra.jsonl",HERE/"selection.jsonl"):
        used.update(r["geometry"] for r in rows(p))
    before=sha256("\n".join(sorted(used)).encode()).hexdigest()
    dest=HERE/"large.jsonl"
    tmp=dest.with_suffix(".building")
    seeds=list(range(9163100,9163220))
    if dest.exists():raise FileExistsError(dest)
    with ProcessPoolExecutor(2) as pool,tmp.open("w",encoding="utf8",newline="\n") as stream:
        for j,batch in enumerate(pool.map(generate,[("spatial-large",s) for s in seeds]),1):
            for r in batch:
                key=geometry(r["grid"])
                assert key==r["geometry"] and key not in used
                used.add(key);stream.write(json.dumps(r,separators=(",",":"))+"\n")
            if j%10==0:print("new geometries",j*100,flush=True)
    tmp.replace(dest)
    save(HERE/"large.json",dict(count=len(seeds)*100,seeds=seeds,sha256=digest(dest),
            excluded_sha256=before,generator_sha256=digest(PACK/"make_hard_maps.py")))


if __name__=="__main__":main()
