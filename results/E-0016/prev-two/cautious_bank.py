"""Независимые источники подгонки, отбора и проверки консервативного решения."""
from concurrent.futures import ProcessPoolExecutor
from .common import HERE,PARENT,TRAIN,DEV,rows,read,write,digest
from .early_bank import FOLDER
from ..data import generate

OUT=HERE/"conservative"
GROUPS={"fit":range(9176420,9176424),"select":range(9176424,9176426),"hold":range(9176426,9176428)}


def bank(group):
    manifest=read(OUT/"registration.json");result=[]
    for seed in GROUPS[group]:
        path=OUT/(str(seed)+".json")
        assert digest(path)==manifest["banks"][str(seed)]
        result.extend(read(path))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    known={r["geometry"] for f in (TRAIN,DEV,PARENT/"data/train.jsonl") for r in rows(f)}
    for label in ("fit1","fit2","fit3","select","hold"):
        known.update(r["geometry"] for r in read(FOLDER/(label+".json")))
    seeds=[s for g in GROUPS.values() for s in g];hashes={}
    with ProcessPoolExecutor(1) as pool:
        for seed,maps in zip(seeds,pool.map(generate,[("prev-two-cautious",s) for s in seeds]),strict=True):
            for r in maps:
                assert r["geometry"] not in known;known.add(r["geometry"])
            path=OUT/(str(seed)+".json");write(path,maps);hashes[str(seed)]=digest(path)
    write(OUT/"registration.json",dict(groups={k:list(v) for k,v in GROUPS.items()},
          banks=hashes,dev=digest(DEV),plan=digest(HERE/"conservative-plan.md"),wave1=digest(HERE/"weights/wave1.pt")))
    print("registered 800",flush=True)


if __name__=="__main__":main()
