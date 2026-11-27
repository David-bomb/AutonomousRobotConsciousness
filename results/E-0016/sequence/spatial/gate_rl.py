"""Отбор сочетания выученных вероятностей выхода по реальной терминальной награде."""
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import torch
from ...data import OUT, digest, read, rows, save
from ...dev import evaluate
from ..belief.population import subset
from . import transfer
from .gate_agent import GateAgent,load
from .model import HERE


def package():
    step=read(HERE/"gate-3000/result.json")["selected"]
    return dict(field_policy=torch.load(HERE.parent/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu"),
        gate=torch.load(HERE/f"gate-3000/s{step}.pt",weights_only=True,map_location="cpu"),
        suffix_archive=torch.from_numpy(np.fromfile(HERE/"suffix-counts.npz",np.uint8)))


def init():
    torch.set_num_threads(1)
    transfer.net=GateAgent(package())


def main():
    out=HERE.parent/"runs/gates3000"
    out.mkdir(exist_ok=True)
    cache=HERE.parent/"cache/physical-gates"
    cache.mkdir(exist_ok=True)
    zero=[1]+[0]*10
    options=[zero]
    for scale in (3,7,13):
        for spec in ((1,1,0,0,0,0,0,0),(1,2,1,0,0,0,0,0),
                     (.5,1,0,0,0,1,1,0),(0,1,0,1,1,0,0,0),
                     (0,0,0,0,0,1,1,0),(.5,1,0,0,0,1,0,.5)):
            options.append([1,*[scale*w for w in spec],-min(scale,3),-min(scale,3)])
    cfg=dict(registration=digest(HERE/"gate-selection.json"),screen=48,selection=200,options=options,
             gate=digest(HERE/"gate-3000/result.json"),suffix=digest(HERE/"suffix.json"),
             foundation=digest(HERE.parent/"runs/field3000-clean/rl.pt"),source=digest(HERE/"gate_rl.py"))
    if (out/"config.json").exists():assert read(out/"config.json")==cfg
    else:save(out/"config.json",cfg)
    # Две серии первых ходов на строго разных полных геометриях.
    with ProcessPoolExecutor(2,initializer=init) as pool:
        screen=transfer.stage(pool,subset(rows(HERE/"large.jsonl")[600:700],12),options,cache/"screen")
        save(out/"screen.json",screen)
        ranked=sorted(screen["rank"],key=lambda r:(-r["wins"],-r["minimum"]))
        finalists=[r["weights"] for r in ranked[:4]]
        if zero not in finalists:finalists.append(zero)
        result=transfer.stage(pool,rows(HERE/"gate-selection.jsonl"),finalists,cache/"selection")
        save(out/"selection.json",result)
    best=max(result["rank"],key=lambda r:(r["wins"],r["minimum"]))
    pack=package()
    pack["linear"]=torch.tensor(best["weights"],dtype=torch.float64)
    path=out/"rl.pt"
    if path.exists():raise FileExistsError(path)
    torch.save(pack,path)
    sig=digest(path);dev=OUT/"data/dev.jsonl"
    save(out/"locked-before-dev.json",dict(weights_sha256=sig,selection=best,dev_sha256=digest(dev)))
    start=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),rows(dev))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(dev),summary=summary,
         episodes=episodes,seconds=time.monotonic()-start))
    print(best,summary,flush=True)


if __name__=="__main__":main()
