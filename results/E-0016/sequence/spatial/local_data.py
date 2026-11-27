"""Ориентированные фрагменты настоящей учебной геометрии для локального прогноза."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from ...data import read,rows,save,digest
from .prepare import LIB,HERE

RADIUS=6


def crop(images,meta):
    offsets=np.arange(-RADIUS,RADIUS+1)
    x=meta[:,0,None]%40+offsets[None,:]
    y=meta[:,0,None]//40+offsets[None,:]
    b=np.arange(len(images))[:,None,None]
    cols=x[:,None,:].clip(0,39);rws=y[:,:,None].clip(0,39)
    outside=(x[:,None,:]<0)|(x[:,None,:]>=40)|(y[:,:,None]<0)|(y[:,:,None]>=40)
    return np.stack((np.where(outside,2,images[b,0,rws,cols]),np.where(outside,0,images[b,1,rws,cols])),1).astype(np.uint8)


def work(job):
    rows_,out=job
    if out.exists():return
    collected=[]
    for r in rows_:
        original=np.array([list(s) for s in r["grid"]],dtype="S1")
        seed=int(r["geometry"][:8],16)
        for j in range(8):
            g=np.rot90(original,j%4)
            if j>=4:g=np.fliplr(g)
            n=12
            x=np.empty((n,2,40,40),np.uint8);meta=np.empty((n,8),np.int16);y=np.empty((n,8,3),np.int8)
            LIB.generate_samples(g.tobytes(),g.shape[1],g.shape[0],n,(seed+103*j)%2**32,x,meta,y)
            collected.append((crop(x,meta),meta,y))
    with out.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f,**dict(zip(("patch","meta","y"),map(np.concatenate,zip(*collected)))))
    out.with_suffix(".tmp").replace(out)


def main():
    source=HERE/"large.jsonl"
    bank=rows(source)
    cache=HERE.parent/"cache/local";cache.mkdir(exist_ok=True)
    jobs=[(bank[i:i+40],cache/f"large-{i:05}.npz") for i in range(0,len(bank),40)]
    with ProcessPoolExecutor(2) as pool:
        for j,_ in enumerate(pool.map(work,jobs),1):
            if j%25==0:print("local maps",40*j,flush=True)
    save(HERE/"local-data.json",dict(source=digest(source),maps=len(bank),states=96*len(bank),seed_stride=103,
         parts={p.name:digest(p) for _,p in jobs}))


if __name__=="__main__":main()
