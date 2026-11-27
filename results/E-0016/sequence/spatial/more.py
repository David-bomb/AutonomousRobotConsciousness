from concurrent.futures import ProcessPoolExecutor
from ...data import rows,save,digest
from .prepare import batch,HERE


def main():
    bank=rows(HERE/"extra.jsonl")
    cache=HERE.parent/"cache/spatial"
    jobs=[(bank[i:i+20],cache/f"extra-{i:04}.npz",128) for i in range(0,len(bank),20)]
    with ProcessPoolExecutor(2) as pool:
        for i,_ in enumerate(pool.map(batch,jobs),1):
            if i%25==0:print("extra",i*20,flush=True)
    save(HERE/"extra-data.json",dict(split="extra",geometries=len(bank),states=len(bank)*128,source=digest(HERE/"extra.jsonl"),
         parts={p.name:digest(p) for _,p,_ in jobs}))


if __name__=="__main__":
    main()
