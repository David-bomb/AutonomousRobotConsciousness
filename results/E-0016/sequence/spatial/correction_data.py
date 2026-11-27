"""Сбор оспариваемых ранних решений joint400 с истинными метками только на train."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import rows,read,save,level,digest
from ...policy import MOVES
from ...teacher import Teacher
from .correction_model import Evidence
from .transfer import package
from .model import HERE


def model_pack():
    pack=package()
    data=np.fromfile(HERE/"prefix-counts.bin",np.int32).reshape(-1,4)
    pack["prefix"]=torch.from_numpy(data)
    return pack


def init():
    global net
    torch.set_num_threads(1)
    net=Evidence(model_pack())


def job(task):
    bank,output=task
    if output.exists():return
    items=[]
    for row in bank:
        rng=np.random.default_rng(int(row["geometry"][:12],16))
        env=Environment(level(row))
        expert=Teacher(env.level)
        seen=set();stalled=0
        while env.status==Status.RUNNING and env.steps<100 and not (env.refuels and env.repairs):
            v=env.view
            z,x,ok=net.directions(v)
            a=int(np.where(ok,z,-1e12).argmax())
            best,viable=expert.labels(env)
            assert viable.any()
            if x is not None and v not in seen:
                d,offset=(v.fuel_distance,0) if v.fuel-v.fuel_distance<=v.health-v.repair_distance else (v.repair_distance,2)
                down=np.array([bool(ok[j] and min(ds[destination(v.memory,v.position,m)] for ds in expert.dist[offset:offset+2])<d)
                               for j,m in enumerate(MOVES)])
                items.append((x.astype(np.float16),ok,down,viable,best,a))
                seen.add(v)
            if not viable[a] or stalled>=16:a=int(rng.choice(np.flatnonzero(best)))
            elif rng.random()<.09:a=int(rng.choice(np.flatnonzero(viable)))
            env.step(MOVES[a])
            stalled=stalled+1 if env.view.memory==v.memory else 0
        expert.remaining.cache_clear()
    with output.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f,**{k:np.stack(v) for k,v in zip(("x","ok","down","viable","best","a"),zip(*items))})
    output.with_suffix(".tmp").replace(output)


def main():
    bank=rows(HERE/"extra.jsonl")
    cache=HERE.parent/"cache/correction";cache.mkdir(exist_ok=True)
    records=[]
    for split,selected in (("fit",bank[2000:2600]),("held",bank[3000:3100])):
        tasks=[(selected[i:i+10],cache/f"{split}-{i:04}.npz") for i in range(0,len(selected),10)]
        with ProcessPoolExecutor(2,initializer=init) as pool:
            for j,_ in enumerate(pool.map(job,tasks),1):
                if j%5==0:print(split,j*10,flush=True)
        n=fatal=critical=0
        for _,p in tasks:
            with np.load(p) as v:
                a=v["a"];viable=v["viable"];n+=len(a);fatal+=int((~viable[np.arange(len(a)),a]).sum())
                critical+=int((v["ok"]&~viable).any(1).sum())
        records.append(dict(split=split,maps=len(selected),states=n,default_fatal=fatal,critical=critical,
                       parts={p.name:digest(p) for _,p in tasks}))
    save(HERE/"correction-data.json",dict(sets=records,prefix=digest(HERE/"prefix-counts.bin"),source=digest(HERE/"correction_data.py")))


if __name__=="__main__":main()
