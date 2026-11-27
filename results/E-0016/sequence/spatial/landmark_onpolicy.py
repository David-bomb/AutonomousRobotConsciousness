"""Учебные наблюдения собственной сети непосредственно перед потерей финиша и при восстановлении."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import level,rows,digest,save
from ...policy import MOVES
from ...teacher import Teacher
from .landmarks import HERE,landmarks
from .model import image
from .transfer_model import FullModel


def init():
    global net,weights
    torch.set_num_threads(1)
    pack=torch.load(HERE.parent/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu")
    net=FullModel(pack)
    weights=pack["linear"].numpy()


def job(args):
    records,path=args
    if path.exists():return
    saved=[]
    for row in records:
        env=Environment(level(row));teacher=Teacher(env.level)
        rng=np.random.default_rng(int(row["geometry"][:12],16))
        used=set();stale=0;cache={}
        while env.status==Status.RUNNING and env.steps<120 and not (env.refuels and env.repairs):
            v=env.view
            if v not in cache:
                z,ok=net.directions(v)
                cache[v]=int(np.where(ok,z@weights,-1e12).argmax()),ok
            a,ok=cache[v]
            best,viable=teacher.labels(env)
            assert viable.any()
            rejected=not viable[a]
            if ok.sum()>1 and v not in used:
                board,meta=image(v)
                pair,acts=landmarks(board,meta)
                assert np.array_equal(ok,acts[:,0,0].astype(bool))
                sites=[v.position,*teacher.sites,teacher.level.goal]
                ds=np.zeros((6,6),np.int16)
                for i,ts in enumerate(teacher.dist,1):ds[i]=[ts[p] for p in sites]
                ds[0]=ds[:,0]
                assert ds[0,1:3].min()==v.fuel_distance and ds[0,3:5].min()==v.repair_distance
                signs=np.full((8,5),-1,np.int8)
                for j,m in enumerate(MOVES):
                    if ok[j]:
                        dest=destination(v.memory,v.position,m)
                        signs[j]=1+np.array([d[dest] for d in teacher.dist])-ds[0,1:]
                saved.append((pair.astype(np.float16),acts.astype(np.float16),meta,(v.fuel,v.health),
                    MOVES.index(v.last_move) if v.last_move else 8,ds,signs,viable,best,a))
                used.add(v)
            if rejected or stale>=18:a=int(rng.choice(np.flatnonzero(best)))
            elif rng.random()<.09:a=int(rng.choice(np.flatnonzero(viable)))
            env.step(MOVES[a])
            stale=stale+1 if v.memory==env.view.memory else 0
        teacher.remaining.cache_clear()
    with path.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(v) for k,v in zip(
            ("pair","act","meta","stock","last","length","sign","safe","best","default"),zip(*saved))})
    path.with_suffix(".pending").replace(path)


def main():
    source=HERE/"large.jsonl"
    bank=rows(source)[1000:2200]
    cache=HERE.parent/"cache/landmarks";cache.mkdir(exist_ok=True)
    jobs=[(bank[i:i+10],cache/f"own-{i:04}.npz") for i in range(0,len(bank),10)]
    with ProcessPoolExecutor(3,initializer=init) as pool:
        for j,_ in enumerate(pool.map(job,jobs),1):
            if j%10==0:print("own",j*10,flush=True)
    n=fatal=0
    for _,p in jobs:
        with np.load(p) as d:
            a=d["default"];n+=len(a)
            fatal+=int((~d["safe"][np.arange(len(a)),a]).sum())
    save(HERE/"landmark-own.json",dict(maps=len(bank),range=[1000,2200],states=n,fatal=fatal,
        source=digest(source),policy=digest(HERE.parent/"runs/field3000-clean/rl.pt"),
        parts={p.name:digest(p) for _,p in jobs},program=digest(HERE/"landmark_onpolicy.py"),
        viable_exploration=.09,max_steps=120,maximum_stale=18))


if __name__=="__main__":main()
