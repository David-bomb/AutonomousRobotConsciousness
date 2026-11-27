"""Окончательная величина поправки выбирается по физическим финишам train."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import time
import numpy as np
import torch
from maze.environment import Environment,Status
from ...data import OUT,rows,read,save,level,digest
from ...dev import evaluate
from ...policy import MOVES
from .correction_model import Correction,load
from .correction_data import model_pack
from .model import HERE


def package(step):
    pack=model_pack()
    pack["calibrator"]=torch.load(HERE/f"calibration/s{step}.pt",weights_only=True,map_location="cpu")
    return pack


def init(step):
    global model
    torch.set_num_threads(1)
    model=Correction(package(step))


def trial(job):
    row,wts=job
    env=Environment(level(row))
    coefficients=np.array(wts).T
    choices={}
    out=[]
    for k in range(len(wts)):
        env.reset()
        while env.status==Status.RUNNING:
            v=env.view
            if v not in choices:
                z,ok=model.directions(v)
                choices[v]=np.where(ok[:,None],z@coefficients,-1e12).argmax(0)
            env.step(MOVES[choices[v][k]])
        out.append(dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs))
    return dict(map=row["name"],tier=row["tier"],outcomes=out)


def main():
    raw=read(HERE/"calibration/result.json");step=raw["selected"]
    result=next(r for r in raw["history"] if r["step"]==step)
    picks=[[1,0,0,0,0,0,0,0]]
    seen=set()
    for r in result["candidates"]:
        w=r["weights"]
        if w in picks:continue
        # Второй вариант с теми же действиями на всех проверенных развилках
        # часто совпадает с первым. Ограничиваем числом разных измеренных компромиссов.
        score=(r["gained"],r["lost"],r["changed"])
        if score in seen:continue
        picks.append(w);seen.add(score)
        if len(picks)>=6:break
    bank=rows(HERE/"selection.jsonl")[100:200]
    cache=HERE.parent/"cache/rl-calibrated";cache.mkdir(exist_ok=True)
    out=HERE.parent/"runs/calibrated600";out.mkdir(exist_ok=True)
    cfg=dict(preselection=digest(HERE/"calibration/result.json"),step=step,weights=picks,train_range=[100,200],
             train_sha256=digest(HERE/"selection.jsonl"),source=digest(HERE/"correction_rl.py"))
    if (out/"config.json").exists():assert read(out/"config.json")==cfg
    else:save(out/"config.json",cfg)
    with ProcessPoolExecutor(3,initializer=init,initargs=(step,)) as pool:
        jobs=[pool.submit(trial,(r,picks)) for r in bank if not (cache/(r["name"]+".json")).exists()]
        for i,f in enumerate(as_completed(jobs),1):
            r=f.result();save(cache/(r["map"]+".json"),r)
            if i%10==0:print("terminal",i,len(jobs),flush=True)
    rec=[read(cache/(r["name"]+".json")) for r in bank]
    scores=[]
    for j,w in enumerate(picks):
        win=Counter(r["tier"] for r in rec if r["outcomes"][j]["outcome"]=="SUCCESS")
        scores.append(dict(weights=w,wins=sum(win.values()),minimum=min(win[t] for t in ("H1","H2","H3","H4"))))
    best=max(scores,key=lambda r:(r["wins"],r["minimum"]))
    save(out/"train.json",dict(scores=scores,records=rec))
    pack=package(step);pack["linear"]=torch.tensor(best["weights"],dtype=torch.float64)
    path=out/"rl.pt";torch.save(pack,path);sig=digest(path)
    save(out/"locked-before-dev.json",dict(weights_sha256=sig,selection=best,dev=digest(OUT/"data/dev.jsonl")))
    t=time.monotonic();summary,episodes=evaluate(load(path.read_bytes()),rows(OUT/"data/dev.jsonl"))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,summary=summary,episodes=episodes,
                dev_sha256=digest(OUT/"data/dev.jsonl"),seconds=time.monotonic()-t))
    print(best,summary,flush=True)


if __name__=="__main__":
    main()
