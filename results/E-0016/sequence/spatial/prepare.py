"""Синтетические частичные обзоры только для обучения предсказателя изменения длин."""
import argparse
import ctypes as ct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from ...data import digest,read,rows,save

HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/"synthesis.dll"))
LIB.generate_samples.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_uint32,
        np.ctypeslib.ndpointer(np.uint8),np.ctypeslib.ndpointer(np.int16),np.ctypeslib.ndpointer(np.int8)]
LIB.generate_samples.restype=None


def batch(job):
    bank,dest,n=job
    if dest.exists():return
    arrays=[]
    for r in bank:
        grid=np.array([list(line) for line in r["grid"]],dtype="S1")
        seed=int(r["geometry"][:8],16)
        for j in range(8):
            g=np.rot90(grid,j%4)
            if j>=4:g=np.fliplr(g)
            count=n//8
            x=np.empty((count,2,40,40),np.uint8)
            meta=np.empty((count,8),np.int16)
            y=np.empty((count,8,3),np.int8)
            LIB.generate_samples(g.tobytes(),g.shape[1],g.shape[0],count,(seed+73*j)%2**32,x,meta,y)
            arrays.append((x,meta,y))
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**dict(zip(("maps","meta","y"),map(np.concatenate,zip(*arrays)))))
    dest.with_suffix(".pending").replace(dest)


def main(n):
    folder=HERE.parent/"cache/spatial"
    folder.mkdir(exist_ok=True)
    outputs=[]
    for split,m,count in (("train",n,256),("probe",100,96)):
        source=HERE.parent/(split+".jsonl")
        bank=rows(source)[:m]
        assert all(max(r["width"],r["height"])<=40 for r in bank)
        jobs=[(bank[i:i+20],folder/f"{split}-{i:04}.npz",count) for i in range(0,m,20)]
        with ProcessPoolExecutor(2) as workers:
            for j,_ in enumerate(workers.map(batch,jobs),1):
                if j%10==0:print(split,j*20,flush=True)
        outputs.append(dict(split=split,geometries=m,states=m*count,source=digest(source),
            parts={p.name:digest(p) for _,p,_ in jobs}))
    save(HERE/f"data-{n}.json",dict(sets=outputs,source_sha256={s:digest(HERE/s) for s in (
            "prepare.py","synthesis.cpp","synthesis.dll","region.cpp","region.dll")}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("n",type=int)
    main(p.parse_args().n)
