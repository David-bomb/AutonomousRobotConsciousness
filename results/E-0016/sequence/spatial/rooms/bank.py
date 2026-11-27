import json
import re
from maze.datasets import geometry
from ....data import OUT,digest,generate,read,save
from ..fresh import fingerprints
from .cells import HERE


def main():
    keys=fingerprints();field=re.compile(rb'"geometry"\s*:\s*"([0-9a-f]{64})"')
    previous=read(HERE.parent/"bank-audit.json")
    sources=[(OUT/r["path"],r["sha256"]) for r in previous["banks"]]
    addition=read(HERE.parent/"decisions-bank.json")
    sources.extend((HERE.parent/(n+".jsonl"),r["sha256"]) for n,r in addition["groups"].items())
    for p,sig in sources:
        assert digest(p)==sig
        with p.open("rb") as f:
            for line in f:keys.add(field.search(line).group(1).decode())
    assert len(keys)==addition["combined"]
    before=len(keys);new=[];seeds=[9164500,9164501]
    for seed in seeds:
        for row in generate(("rooms-select",seed)):
            k=geometry(row["grid"]);assert row["geometry"]==k and k not in keys
            keys.add(k);new.append(row)
    dest=HERE/"selection.jsonl"
    with dest.open("x",encoding="utf8",newline="\n") as f:
        for row in new:f.write(json.dumps(row,separators=(",",":"))+"\n")
    save(HERE/"bank.json",dict(maps=len(new),seeds=seeds,excluded=before,shared=0,canonical_sha256=digest(dest),
                  prior_registration=digest(HERE.parent/"decisions-bank.json")))
    print(len(new),before,flush=True)


if __name__=="__main__":main()
