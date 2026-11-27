"""Реальные снимки поездок дополняют упрощённую геометрическую предтренировку."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import rows,read,save,level,digest
from ...policy import MOVES
from ...teacher import Teacher
from ..alt_model import ExitNet,exits
from ..model import encode
from .model import HERE,image


def init():
    global net,prior
    torch.set_num_threads(1)
    net=ExitNet().eval()
    net.load_state_dict(torch.load(HERE.parent/"runs/exit600/s900.pt",weights_only=True,map_location="cpu"))
    prior=net.wall_prior.numpy().tobytes()


@torch.inference_mode()
def batch(job):
    bank,traces,path=job
    if path.exists():return
    saved=[]
    for row,seq in zip(bank,traces):
        rng=np.random.default_rng(int(row["geometry"][:12],16))
        env=Environment(level(row))
        teacher=Teacher(env.level)
        seen=set()
        stalled=0
        while env.status==Status.RUNNING and env.steps<(len(seq) if seq is not None else 220):
            v=env.view
            best,viable=teacher.labels(env)
            branching=sum(destination(v.window,(1,1),m) is not None for m in MOVES)>1
            if seq is not None:
                a=default=int(seq[env.steps])
            else:
                x,ok=encode(v,prior)
                ex=exits(v,prior)
                raw=net.full(torch.from_numpy(x[None]),torch.from_numpy(ex[None]),torch.from_numpy(ok[None]))[0]
                a=default=int(raw.argmax())
                if not viable[a] or stalled>=14:
                    assert best.any()
                    a=int(rng.choice(np.flatnonzero(best)))
            if branching and v not in seen and (seq is None or rng.random()<.7):
                img,meta=image(v)
                truth=np.full((8,3),-1,np.int8)
                for j,m in enumerate(MOVES):
                    q=destination(v.memory,v.position,m)
                    if q is not None:
                        truth[j]=1+np.array([min(d[q] for d in teacher.dist[:2])-v.fuel_distance,
                           min(d[q] for d in teacher.dist[2:4])-v.repair_distance,teacher.dist[4][q]-teacher.dist[4][v.position]])
                assert truth.max()<=2
                saved.append((img,meta,truth,viable,default,[v.fuel-v.fuel_distance,v.health-v.repair_distance]))
                seen.add(v)
            env.step(MOVES[a])
            stalled=stalled+1 if v.memory==env.view.memory else 0
        teacher.remaining.cache_clear()
    with path.with_suffix(".temp").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(col) for k,col in zip(
            ("maps","meta","y","viable","default","slack"),zip(*saved))})
    path.with_suffix(".temp").replace(path)


def main():
    traces={r["map"]:r["actions"] for r in read(HERE.parent/"clean-400.json")["episodes"]}
    groups=[]
    cache=HERE.parent/"cache/spatial"
    with ProcessPoolExecutor(2,initializer=init) as pool:
        for split,n in (("train",400),("probe",100)):
            bank=rows(HERE.parent/(split+".jsonl"))[:n]
            jobs=[(bank[i:i+10],[traces[r["name"]] if split=="train" else None for r in bank[i:i+10]],
                   cache/f"real-{split}-{i:04}.npz") for i in range(0,n,10)]
            for j,_ in enumerate(pool.map(batch,jobs),1):
                if j%5==0:print("real",split,j*10,flush=True)
            groups.append(dict(split=split,count=n,parts={p.name:digest(p) for _,_,p in jobs}))
    save(HERE/"real.json",dict(sets=groups,source_sha256=digest(HERE/"real.py"),annotations="F,R,B signs and train reachability"))


if __name__=="__main__":
    main()
