from concurrent.futures import ProcessPoolExecutor
import time
import torch
from ...data import OUT,read,rows,save,digest
from ...dev import evaluate
from ..belief.population import subset
from . import transfer
from .model import HERE
from .route_agent import RouteAgent,load


def package():
    chosen=read(HERE/"landmark-own-fit/result.json")["selected"]
    return dict(foundation=torch.load(HERE.parent/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu"),
                routes=torch.load(HERE/f"landmark-own-fit/s{chosen}.pt",weights_only=True,map_location="cpu"))


def init():
    torch.set_num_threads(1)
    transfer.net=RouteAgent(package())


def main():
    zero=[1]+[0]*9
    prior=[1,4,2,2,2,0,2,0,0,0]
    options=[zero,prior]
    for k in (2,6,12):
        for recipe in ((1,0,1,0,0,0,0,0,0),(1,0,1,1,1,0,0,0,0),
                (.5,.5,0,1,0,1,1,0,0),(0,0,0,0,0,0,1,.5,0),
                (0,0,0,1,1,0,0,.5,.25),(0,0,0,0,0,0,0,1,0)):
            options.append([1,*[v*k for v in recipe]])
    out=HERE.parent/"runs/route1200";out.mkdir(exist_ok=True)
    cache=HERE.parent/"cache/physical-route";cache.mkdir(exist_ok=True)
    cfg=dict(train=digest(HERE/"route-selection.jsonl"),options=options,pre_RL=prior,
        training=digest(HERE/"landmark-own-fit/result.json"),source=digest(HERE/"route_rl.py"))
    if (out/"config.json").exists():assert read(out/"config.json")==cfg
    else:save(out/"config.json",cfg)
    with ProcessPoolExecutor(3,initializer=init) as pool:
        screen=transfer.stage(pool,subset(rows(HERE/"large.jsonl")[4000:4100],12),options,cache/"screen")
        save(out/"screen.json",screen)
        ranked=sorted(screen["rank"],key=lambda r:(-r["wins"],-r["minimum"]))
        choices=[r["weights"] for r in ranked[:4]]
        for w in (zero,prior):
            if w not in choices:choices.append(w)
        result=transfer.stage(pool,rows(HERE/"route-selection.jsonl"),choices,cache/"selection")
        save(out/"selection.json",result)
    best=max(result["rank"],key=lambda r:(r["wins"],r["minimum"]))
    dev=OUT/"data/dev.jsonl";packs=[]
    for name,w in (("rl",best["weights"]),("pre",prior)):
        pack=package();pack["linear"]=torch.tensor(w,dtype=torch.float64)
        path=out/f"{name}.pt"
        if path.exists():raise FileExistsError(path)
        torch.save(pack,path);packs.append(path)
    save(out/"locked-before-dev.json",dict(weights_sha256=digest(packs[0]),pre_sha256=digest(packs[1]),
                 selection=best,dev_sha256=digest(dev)))
    for p in packs:
        start=time.monotonic()
        s,episodes=evaluate(load(p.read_bytes()),rows(dev))
        save(p.with_suffix(".physical.json"),dict(weights_sha256=digest(p),dev_sha256=digest(dev),summary=s,
                    episodes=episodes,seconds=time.monotonic()-start))
        print(p.name,s,flush=True)


if __name__=="__main__":main()
