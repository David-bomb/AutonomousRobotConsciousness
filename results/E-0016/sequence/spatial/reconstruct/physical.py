from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

from ....data import rows, read, save, digest
from ....dev import evaluate
from .. import transfer
from ..rooms.physical import summarize
from .adapter import load
from .bank import HERE, RUN, CACHE, PACK, DEV, maps
from .model import Policy

MODES = {"marginal1":[0,1,0], "direct1":[0,0,1], "missing":[1,0,0],
         "marginal0.5":[.5,.5,0], "direct0.5":[.5,0,.5]}


def pack(mode):
    return dict(foundation=torch.load(PACK, weights_only=True, map_location="cpu"),
                decision=torch.load(HERE/"decision.pt",weights_only=True,map_location="cpu"),
                mode=MODES[mode], method=mode)


def init_selection():
    torch.set_num_threads(1)
    transfer.net=Policy(pack("marginal1"))


def init_dev(files):
    global controllers
    controllers={name:load(path.read_bytes()) for name,path in files.items()}


def trip(job):
    name,row=job
    return name,evaluate(controllers[name],[row])[1][0]


def main():
    cfg=dict(registration=digest(HERE/"registration.json"), weights=digest(HERE/"decision.pt"),
             held=digest(HERE/"held.json"), code={n:digest(HERE/n) for n in ("model.py","features.py","adapter.py","physical.py")})
    if (RUN/"config.json").exists():
        assert read(RUN/"config.json")==cfg
    else:
        save(RUN/"config.json",cfg)
    labels=read(HERE/"registration.json")["selection_order"]
    assert labels==list(MODES)
    with ProcessPoolExecutor(4,initializer=init_selection) as pool:
        selection=transfer.stage(pool,maps("selection"),[MODES[k] for k in labels],CACHE/"selection")
    save(RUN/"selection.json",selection)
    rank={label:r for label,r in zip(labels,selection["rank"],strict=True)}
    def winner(names):
        return max(names,key=lambda k:(rank[k]["wins"],rank[k]["minimum"]))
    methods={name:winner([k for k in labels if k.startswith(name)]) for name in ("missing","marginal","direct")}
    choice=winner(labels)
    files={}
    for name,method in methods.items():
        dest=RUN/(name+".pt")
        if not dest.exists():
            torch.save(pack(method),dest)
        assert torch.load(dest,weights_only=True)["method"]==method
        files[name]=dest
    chosen=next(k for k,v in methods.items() if v==choice)
    best=RUN/"rl.pt"
    if not best.exists():
        best.write_bytes(files[chosen].read_bytes())
    assert digest(best)==digest(files[chosen])
    lock=dict(methods=methods,chosen=chosen,rank=rank,hashes={n:digest(p) for n,p in files.items()},
              weights_sha256=digest(best),dev=digest(DEV),selection=digest(RUN/"selection.json"))
    if (RUN/"locked-before-dev.json").exists():
        assert read(RUN/"locked-before-dev.json")==lock
    else:
        save(RUN/"locked-before-dev.json",lock)
    print("frozen",lock,flush=True)
    bank=rows(DEV)
    for name in files:
        (CACHE/("dev-"+name)).mkdir(exist_ok=True)
    jobs=[(name,row) for row in bank for name in files if not (CACHE/("dev-"+name)/(row["name"]+".json")).exists()]
    with ProcessPoolExecutor(4,initializer=init_dev,initargs=(files,)) as pool:
        pending=[pool.submit(trip,job) for job in jobs]
        for i,fut in enumerate(as_completed(pending),1):
            name,ep=fut.result()
            save(CACHE/("dev-"+name)/(ep["map"]+".json"),ep)
            if i%50==0:print("dev",i,len(jobs),flush=True)
    for name in files:
        episodes=[read(CACHE/("dev-"+name)/(r["name"]+".json")) for r in bank]
        data=dict(weights_sha256=lock["hashes"][name],dev_sha256=digest(DEV),
                  summary=summarize(episodes),episodes=episodes)
        save(files[name].with_suffix(".physical.json"),data)
        if name==chosen:
            save(best.with_suffix(".physical.json"),data)
        print(name,data["summary"],flush=True)


if __name__=="__main__":
    main()
