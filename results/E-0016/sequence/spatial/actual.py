"""Трудные собственные учебные поездки с физическим исправлением на train."""
from concurrent.futures import ProcessPoolExecutor
from ...data import rows,save,digest
from .real import init,batch
from .model import HERE


def main():
    source=HERE.parent/"train.jsonl"
    bank=rows(source)[400:800]
    cache=HERE.parent/"cache/spatial"
    jobs=[(bank[i:i+10],[None]*10,cache/f"actual-{i:04}.npz") for i in range(0,len(bank),10)]
    with ProcessPoolExecutor(2,initializer=init) as pool:
        for j,_ in enumerate(pool.map(batch,jobs),1):
            if j%5==0:print("actual",j*10,flush=True)
    save(HERE/"actual.json",dict(split="actual",parts={p.name:digest(p) for _,_,p in jobs},
            train_sha256=digest(source),train_range=[400,800],representation=digest(HERE/"real.py")))


if __name__=="__main__":
    main()
