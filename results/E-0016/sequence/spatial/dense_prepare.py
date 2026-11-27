import ctypes as ct
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from ...data import read,rows,save,digest
from .model import HERE

NATIVE=ct.CDLL(str(HERE/"dense.dll"))
NATIVE.label_dense.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.int16),
                              np.ctypeslib.ndpointer(np.uint8)]
NATIVE.label_dense.restype=ct.c_int


def label(job):
    bank,original,out=job
    if out.exists():return
    with np.load(original) as f:
        meta=f["meta"]
    each=len(meta)//len(bank)//8
    result=np.zeros((len(meta),40,40),np.uint8)
    for i,r in enumerate(bank):
        g=np.array([list(s) for s in r["grid"]],dtype="S1")
        for j in range(8):
            grid=np.rot90(g,j%4)
            if j>=4:grid=np.fliplr(grid)
            sl=slice((i*8+j)*each,(i*8+j+1)*each)
            assert not NATIVE.label_dense(grid.tobytes(),grid.shape[1],grid.shape[0],each,meta[sl],result[sl])
    with out.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,dense=result)
    out.with_suffix(".pending").replace(out)


def main():
    bank=rows(HERE/"extra.jsonl")
    originals=read(HERE/"extra-data.json")
    folder=HERE.parent/"cache/spatial"
    jobs=[(bank[j*20:j*20+20],folder/n,folder/("dense-"+n)) for j,n in enumerate(originals["parts"])]
    with ProcessPoolExecutor(2) as workers:
        for i,_ in enumerate(workers.map(label,jobs),1):
            if i%30==0:print("dense",i*20,flush=True)
    save(HERE/"dense.json",dict(original=digest(HERE/"extra-data.json"),parts={o.name:digest(o) for _,_,o in jobs}))


if __name__=="__main__":
    main()
