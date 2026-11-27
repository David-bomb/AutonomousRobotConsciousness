"""На train разделяются эффект первого изменения и повторное применение головы по всей поездке."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment,Status
from ...data import digest,read,rows,save,level
from .model import HERE,MOVES
from .terminal_model import TerminalModel


def init():
    global model,weights
    torch.set_num_threads(1)
    pack=torch.load(HERE.parent/"runs/paired200-observed/rl.pt",weights_only=True,map_location="cpu")
    model=TerminalModel(pack);weights=pack["linear"].numpy()


def trip(args):
    row,path=args
    env=Environment(level(row));cache={};result={}
    for mode in ("original","first_change","all_changes","slack01"):
        env.reset();changes=[0,0,0]
        while env.status==Status.RUNNING:
            v=env.view
            if v not in cache:
                x,ok=model.directions(v)
                cache[v]=(int(np.where(ok,x[:,0],-1e12).argmax()),int(np.where(ok,x@weights,-1e12).argmax()))
            a,b=cache[v]
            if mode=="original" or (mode=="first_change" and sum(changes)) or (
                    mode=="slack01" and min(v.fuel-v.fuel_distance,v.health-v.repair_distance) not in (0,1)):
                pick=a
            else:pick=b
            if pick!=a:changes[int(bool(env.refuels))+int(bool(env.repairs))]+=1
            env.step(MOVES[pick])
        result[mode]=dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,changes=changes)
    save(path,dict(map=row["name"],tier=row["tier"],modes=result))


def main():
    cache=HERE.parent/"cache/paired-diagnosis";cache.mkdir(exist_ok=True)
    src=HERE/"reward-screen.jsonl"
    jobs=[(r,cache/(r["name"]+".json")) for r in rows(src)]
    with ProcessPoolExecutor(2,initializer=init) as pool:
        pending=[pool.submit(trip,j) for j in jobs if not j[1].exists()]
        for i,f in enumerate(as_completed(pending),1):
            f.result()
            if i%20==0:print(i,len(pending),flush=True)
    records=[read(p) for _,p in jobs];summary={}
    for name in records[0]["modes"]:
        modes=[r["modes"][name] for r in records]
        summary[name]=dict(outcomes=dict(Counter(r["outcome"] for r in modes)),
            both=sum(bool(r["refuels"] and r["repairs"]) for r in modes),changed_actions=np.sum([r["changes"] for r in modes],0).tolist())
    save(HERE/"paired-diagnosis.json",dict(train_sha256=digest(src),source=digest(HERE/"paired_diagnosis.py"),summary=summary,records=records))
    print(summary,flush=True)


if __name__=="__main__":main()
