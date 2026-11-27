"""Разметка подражания из реально исполненных поездок учителя на train."""
import os
for setting in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[setting]="1"

from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
from maze.environment import Environment, Status
from neural_rl.perception import immediate
from .bank import HERE, digest, level, read, save, train_order
from .features import MOVES, encode


def prepare(job):
    row,kind=job
    name=row["name"]
    trace_file=HERE/f"{kind}-traces"/(name+".json")
    trace=read(trace_file)
    assert trace["map"]==name and trace["geometry"]==row["geometry"]
    folder=HERE/f"{kind}-samples"
    folder.mkdir(exist_ok=True)
    path=folder/(name+".npz")
    stamp=digest(trace_file)
    sources=tuple(digest(HERE/s) for s in ("dataset.py","features.py","fronts.dll"))
    if path.exists():
        with np.load(path) as data:
            assert data["trace"].item()==stamp and tuple(data["sources"].tolist())==sources
            return name,int(len(data["actions"])),digest(path)
    env=Environment(level(row))
    unique={}
    end=(len(trace["actions"]) if trace["cycle_start"] is None else
         trace["prefix"]+trace["cycle_start"]+trace["cycle_len"])
    for step,code in enumerate(trace["actions"][:end]):
        assert env.status==Status.RUNNING
        action=int(code)
        view=env.view
        if step>=trace["prefix"] and view not in unique:
            legal=immediate(view.window)
            assert legal[action]
            phase=int(env.refuels>0)+int(env.repairs>0)
            unique[view]=(action,phase,int(legal.sum()))
        env.step(MOVES[action])
        assert env.moved
    samples=list(unique)
    if len(samples)>320:
        rng=np.random.default_rng(int(row["geometry"][:12],16))
        importance=np.array([(1+.5*(phase==2))*(2 if degree>=3 else .2 if degree==1 else 1)
                             for _,phase,degree in unique.values()])
        chosen=rng.choice(len(samples),320,replace=False,p=importance/importance.sum())
        samples=[samples[i] for i in sorted(chosen)]
    edges,tokens,masks,actions,phases=[],[],[],[],[]
    for view in samples:
        edge,front,mask=encode(view)
        action,phase,_=unique[view]
        edges.append(edge); tokens.append(front); masks.append(mask)
        actions.append(action); phases.append(phase)
    # Пустые корректирующие поездки учитываются в журнале и пропускаются при загрузке.
    pending=path.with_suffix(".pending.npz")
    np.savez_compressed(pending,edges=np.array(edges),tokens=np.array(tokens),legal=np.array(masks),
                        actions=np.array(actions,np.int64),phase=np.array(phases,np.uint8),
                        win=trace["outcome"]=="SUCCESS",trace=stamp,sources=sources)
    pending.replace(path)
    return name,len(samples),digest(path)


def build(kind):
    manifest=read(HERE/(kind+".json"))
    bank={r["name"]:r for r in train_order()}
    jobs=[(bank[name],kind) for name in manifest["files"]]
    with ProcessPoolExecutor(2) as workers:
        rows=[]
        for item in workers.map(prepare,jobs):
            rows.append(item)
            if len(rows)%64==0:
                print(kind,"encoded",len(rows),"/",len(jobs),flush=True)
    save(HERE/(kind+"-samples.json"),dict(records=sum(r[1] for r in rows),
         maps=len(rows),files={name:sha for name,n,sha in rows},nonempty=sum(r[1]>0 for r in rows)))


if __name__=="__main__":
    build(sys.argv[1])
