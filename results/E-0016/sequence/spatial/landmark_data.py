"""Строго учебные длины между отдельными станциями и метки для каждого первого действия."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from maze.grid import wave
from ...data import digest,read,rows,save
from ...policy import MOVES
from .gate_data import site_key
from .landmarks import landmarks,HERE


def job(args):
    bank,source,out=args
    if out.exists():return
    grids={site_key(tuple(r["grid"])):tuple(r["grid"]) for r in bank}
    @lru_cache(maxsize=128)
    def lengths(key,target):
        d=np.full(1600,300,np.int16)
        for (x,y),steps,_ in wave(grids[key],[(target%40,target//40)]):
            d[y*40+x]=steps
        return d
    rec=[]
    with np.load(source) as data:
        boards,metas,stocks,lasts,ys,viable,fast=[data[k] for k in ("maps","meta","stocks","last","y","viable","best")]
    for board,m,stocks_,last,y,success,best in zip(boards,metas,stocks,lasts,ys,viable,fast):
        pair,act=landmarks(board,m)
        ok=act[:,0,0]>0
        assert np.array_equal(ok,y[:,0]>=0)
        ends=np.stack([lengths(tuple(m[1:6]),t) for t in m[1:6]])
        ds=np.zeros((6,6),np.int16)
        ds[1:,:]=ends[:,m[:6]];ds[0,:]=ds[:,0]
        assert ds[0,1:3].min()==m[6] and ds[0,3:5].min()==m[7]
        assert np.all(np.minimum(ds,127)>=np.rint(pair[:,:,0]*127))
        assert np.all(np.minimum(ds,127)<=np.rint(pair[:,:,1]*127))
        neigh=m[0]+np.array([40*a.value[1]+a.value[0] for a in MOVES])
        signs=np.full((8,5),-1,np.int8)
        signs[ok]=1+ends[:,neigh[ok]].T-ds[0,1:]
        assert signs.min()>=-1 and signs.max()<=2
        rec.append((pair.astype(np.float16),act.astype(np.float16),m,stocks_,last,
                    ds,signs,success,best))
    with out.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(c) for k,c in zip(("pair","act","meta","stock","last","length","sign","safe","best"),zip(*rec))})
    out.with_suffix(".pending").replace(out)


def main():
    folder=HERE.parent/"cache/landmarks"
    folder.mkdir(exist_ok=True)
    bank=rows(HERE/"extra.jsonl")
    groups=[]
    for g,cohort in zip(read(HERE/"early.json")["sets"],(bank[:3000],bank[3000:])):
        jobs=[(cohort[20*i:20*i+20],HERE.parent/"cache/spatial"/n,folder/n) for i,n in enumerate(g["parts"])]
        with ProcessPoolExecutor(2) as pool:
            for j,_ in enumerate(pool.map(job,jobs),1):
                if j%25==0:print(g["split"],j*20,flush=True)
        groups.append(dict(maps=len(cohort),split=g["split"],parts={p.name:digest(p) for _,_,p in jobs}))
    save(HERE/"landmark-data.json",dict(sets=groups,source=digest(HERE/"early.json"),features=digest(HERE/"landmarks.cpp"),
          labels=digest(HERE/"landmark_data.py"),nodes=["position","F1","F2","R1","R2","B"]))


if __name__=="__main__":main()
