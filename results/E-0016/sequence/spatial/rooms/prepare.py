from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from ....data import read,rows,save,digest
from ..gate_data import site_key
from ..gates import describe
from .cells import HERE,layout,at,representation

FAMILIES=("corridors","branching","loops","rooms")


def full_map(row):
    grid=np.array([list(s) for s in row["grid"]])
    return grid!="#"


def counts(bank):
    lookup={}
    for row in bank:
        if row["family"]!="rooms":continue
        w,h=row["width"],row["height"];real=full_map(row)
        c,edges,_,_,even,_,_=layout(w,h)
        marked=at(real,edges).any(1)
        assert at(real,c)[marked].all()
        mask=np.zeros((h,w),bool);xy=edges[marked];mask[xy[...,1],xy[...,0]]=True
        assert np.array_equal(mask&even,real&even)
        lookup.setdefault((h,w),[]).append(int(marked.sum()))
    table=np.zeros((40,40,2),np.float32)
    hw=np.array(list(lookup))
    for h in range(19,38,2):
        for w in range(19,38,2):
            group=lookup.get((h,w),[])+lookup.get((w,h),[])
            if not group:
                k=tuple(hw[np.argmin(((hw-[h,w])**2).sum(1))]);group=lookup[k]
                table[h,w]=np.mean(group),len(layout(w,h)[0])
            else:
                table[h,w]=np.mean(group),max(group)
    return table,len(lookup)


def shard(args):
    bank,source,dest,table,hold=args
    if dest.exists():return
    layouts={site_key(tuple(r["grid"])):(r,full_map(r)) for r in bank}
    rec=[]
    with np.load(source) as f:
        maps,metas,stocks,lasts,ys,safe,best=[f[k] for k in ("maps","meta","stocks","last","y","viable","best")]
    groups={}
    for i,m in enumerate(metas):groups.setdefault(tuple(m[1:6]),[]).append(i)
    for key,idx in groups.items():
        row,truth=layouts[key];family=FAMILIES.index(row["family"])
        w,h=row["width"],row["height"]
        # Равномерное покрытие пути, одинаковая верхняя граница числа примеров с карты.
        selected=np.array(idx)[np.linspace(0,len(idx)-1,min(12 if hold else 8,len(idx))).astype(int)]
        for order,i in enumerate(selected):
            orig,m,s=maps[i],metas[i],stocks[i]
            hypo,state,added,blocked=representation(orig,m,w,h,table)
            check=[added.sum(),(added&truth).sum(),blocked.sum(),(blocked&~truth).sum()]
            if family==3:assert check[0]==check[1] and check[2]==check[3]
            x,_,u=describe(orig,m,s)
            z=x if hypo is orig else describe(hypo,m,s)[0]
            ok=ys[i,:,0]>=0
            rec.append((x.astype(np.float16),z.astype(np.float16),state,check,ok,ys[i,:,u]==0,safe[i],best[i],
                        family,int(row["tier"][1]),order==0,(s-m[6:]).min(),lasts[i],s,m))
    with dest.with_suffix(".pending").open("wb") as f:
        names=("old","room","state","check","legal","down","safe","best","family","tier","first","slack","last","stock","meta")
        np.savez_compressed(f,**{k:np.stack(v) for k,v in zip(names,zip(*rec))})
    dest.with_suffix(".pending").replace(dest)


def main():
    parent=HERE.parent
    bank=rows(parent/"extra.jsonl")
    table,n=counts(bank[:3000])
    cache=parent.parent/"cache/rooms";cache.mkdir(exist_ok=True)
    specs=read(parent/"early.json")["sets"];groups=[]
    for group,part in zip(specs,(bank[:3000],bank[3000:])):
        is_held=group["split"]=="early-held"
        jobs=[(part[20*i:20*i+20],parent.parent/"cache/spatial"/name,cache/name,table,is_held)
               for i,name in enumerate(group["parts"])]
        with ProcessPoolExecutor(2) as pool:
            for j,_ in enumerate(pool.map(shard,jobs),1):
                if j%40==0:print(group["split"],j*20,flush=True)
        summary={}
        for _,_,p,_,_ in jobs:
            with np.load(p) as f:
                for j,name in enumerate(FAMILIES):
                    use=f["family"]==j
                    counts0=np.r_[use.sum(),f["check"][use].sum(0)].astype(int)
                    summary[name]=(np.asarray(summary.get(name,[0]*5))+counts0).tolist()
        groups.append(dict(split=group["split"],geometries=len(part),by_family=summary,parts={j[2].name:digest(j[2]) for j in jobs}))
    with (HERE/"counts.npy").open("wb") as f:np.save(f,table)
    save(HERE/"data.json",dict(groups=groups,families=FAMILIES,count_training_maps=3000,
        original_size_pairs=n,source=digest(parent/"extra.jsonl"),registered=digest(parent/"early.json"),
        count_sha256=digest(HERE/"counts.npy"),program=digest(HERE/"prepare.py"),representation=digest(HERE/"cells.py"),
        fields=["states","inferred_core_roads","actually_road","inferred_walls","actually_wall"]))


if __name__=="__main__":main()
