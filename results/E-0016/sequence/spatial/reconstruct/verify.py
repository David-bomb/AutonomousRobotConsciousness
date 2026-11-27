"""Интерфейс замороженной односнимковой политики и сверка обычных поездок."""
from importlib import import_module
from pathlib import Path
import sys
import numpy as np

from maze.environment import Environment, Status
from ....data import OUT, ROOT, digest, read, save, level
from ...publish_next import PACKAGES
from .bank import DEV, HERE, RUN, maps


def check(folder, name, kind):
    adapter, sources=PACKAGES[kind]
    module=import_module("results.E-0016.sequence."+adapter[:-3].replace("/","."))
    weights=folder/(name+".pt")
    record=read(weights.with_suffix(".physical.json"))
    assert record["weights_sha256"]==digest(weights) and record["dev_sha256"]==digest(DEV)
    data=weights.read_bytes()
    policy=module.load(data)
    assert not hasattr(policy,"previous")
    snapshots=[]
    fit=maps("fit")
    for tier in ("H1","H2","H3","H4"):
        row=next(r for r in fit if r["tier"]==tier)
        env=Environment(level(row))
        for _ in range(18):
            if env.status!=Status.RUNNING:break
            v=env.view;move=policy.choose(v)
            snapshots.append((v,move))
            env.step(move)
    cold=module.load(data)
    for index in np.random.default_rng(109).permutation(len(snapshots)):
        v,move=snapshots[index]
        assert cold.choose(v)==move
    verifier=ROOT/"results/verification-rl80"
    sys.path.insert(0,str(verifier))
    from evaluate import episode
    from suite import inventory
    cases,_=inventory([DEV],"validation")
    isolated=episode(cases[0],(HERE.parent.parent/adapter).read_bytes(),data)
    assert all(isolated[k]==record["episodes"][0][k] for k in ("outcome","steps","refuels","repairs"))
    shared=("policy.py","rich_policy.py","critical_policy.py","geometry.dll","geometry.cpp","dev.py")
    paths=[OUT/s for s in shared]+[HERE.parent.parent/s for s in ("model.py","geometry.cpp","sequence.dll",adapter,*sources)]
    spec=dict(weights_sha256=digest(weights),adapter=(HERE.parent.parent/adapter).relative_to(OUT).as_posix(),
              dev_sha256=digest(DEV),full_episodes=len(record["episodes"]),reordered_training_snapshots=len(snapshots),
              isolated=isolated,worker_sha256=digest(verifier/"policy_worker.py"),summary=record["summary"],
              sources={p.resolve().relative_to(OUT).as_posix():digest(p) for p in paths})
    save(weights.with_suffix(".verified.json"),spec)
    print(name,len(snapshots),isolated["outcome"],isolated["steps"],flush=True)


if __name__=="__main__":
    folder=RUN if sys.argv[1]=="primary" else RUN.parent/"reconstruct-dagger200"
    kind="reconstruct" if sys.argv[1]=="primary" else "reconstruct-dagger"
    for tag in sys.argv[2:]:
        check(folder,tag,kind)
