"""Новые источники для второго консервативного опыта."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from . import cautious_collect,early_data
from .cautious_bank import bank as first_bank,OUT as FIRST
from .common import HERE,PARENT,DEV,TRAIN,read,write,rows,digest
from .early_bank import FOLDER
from ..data import generate

OUT=HERE/"geodesic"
GROUPS={"fit":range(9176440,9176446),"select":range(9176446,9176448),"hold":range(9176448,9176450)}


def bank(group):
    reg=read(OUT/"registration.json");maps=[]
    for seed in GROUPS[group]:
        file=OUT/(str(seed)+".json");assert digest(file)==reg["banks"][str(seed)]
        maps.extend(read(file))
    return maps


def collect(row):
    cautious_collect.OUT=OUT
    return cautious_collect.gather(row)


def main():
    known={r["geometry"] for f in (DEV,TRAIN,PARENT/"data/train.jsonl") for r in rows(f)}
    for g in ("fit","select","hold"):known.update(r["geometry"] for r in first_bank(g))
    for name in ("fit1","fit2","fit3","select","hold"):known.update(r["geometry"] for r in read(FOLDER/(name+".json")))
    OUT.mkdir(exist_ok=True);seeds=[s for group in GROUPS.values() for s in group]
    hashes={}
    with ProcessPoolExecutor(2) as pool:
        for s,maps in zip(seeds,pool.map(generate,[("prev-two-geodesic",s) for s in seeds]),strict=True):
            for r in maps:assert r["geometry"] not in known;known.add(r["geometry"])
            path=OUT/(str(s)+".json");write(path,maps);hashes[str(s)]=digest(path)
    fit=[r for s in GROUPS["fit"] for r in read(OUT/(str(s)+".json"))]+first_bank("fit")
    wall=[0]*5;total=[0]*5
    for row in fit:
        grid=row["grid"];h=len(grid);w=len(grid[0])
        for y,line in enumerate(grid):
            for x,c in enumerate(line):
                i=4 if y in (0,h-1) or x in (0,w-1) else y%2*2+x%2
                total[i]+=1;wall[i]+=int(c=="#")
    write(OUT/"registration.json",dict(groups={k:list(v) for k,v in GROUPS.items()},banks=hashes,
          plan=digest(HERE/"geodesic-plan.md"),wall_prior=[a/b for a,b in zip(wall,total)],counts=total))
    (OUT/"prefix").mkdir(exist_ok=True)
    with ProcessPoolExecutor(2,initializer=early_data.init,initargs=("wave1",)) as pool:
        book={}
        for name,h in pool.map(collect,bank("fit")):
            book[name]=h
            if len(book)%100==0:print("geodesic prefixes",len(book),flush=True)
    write(OUT/"prefixes.json",dict(files=book))


if __name__=="__main__":main()
