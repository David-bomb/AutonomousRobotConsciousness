from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from ....data import read,save,rows,digest
from ....dev import evaluate
from .. import transfer
from ..rooms.physical import summarize
from .bank import HERE, CACHE, RUN, DEV
from .dagger import extra
from .dagger_model import Comparison
from .dagger_adapter import load

OUT=RUN.parent/"reconstruct-dagger200"
NAMES=("previous","marginal","direct","probability")


def init_train():
    torch.set_num_threads(1)
    transfer.net=Comparison(torch.load(RUN/"rl.pt",weights_only=True),
                            torch.load(HERE/"dagger.pt",weights_only=True))


def init_dev():
    global ctrl
    ctrl=load((OUT/"rl.pt").read_bytes())


def journey(row):
    return evaluate(ctrl,[row])[1][0]


def main():
    OUT.mkdir(exist_ok=True)
    cfg=dict(registration=digest(HERE/"dagger-registration.json"), previous=digest(RUN/"rl.pt"),
             updated=digest(HERE/"dagger.pt"),sources={n:digest(HERE/n) for n in
                ("dagger.py","dagger_model.py","dagger_adapter.py","dagger_physical.py")})
    assert read(RUN/"locked-before-dev.json")["weights_sha256"]==cfg["previous"]
    if (OUT/"config.json").exists():assert read(OUT/"config.json")==cfg
    else:save(OUT/"config.json",cfg)
    bank=extra("selection")
    with ProcessPoolExecutor(4,initializer=init_train) as pool:
        sel=transfer.stage(pool,bank,np.eye(4).tolist(),CACHE/"dagger-selection")
    save(OUT/"selection.json",sel)
    idx=max(range(4),key=lambda i:(sel["rank"][i]["wins"],sel["rank"][i]["minimum"]))
    path=OUT/"rl.pt"
    if not path.exists():
        if idx==0:path.write_bytes((RUN/"rl.pt").read_bytes())
        else:
            pack=torch.load(RUN/"rl.pt",weights_only=True)
            pack.update(decision=torch.load(HERE/"dagger.pt",weights_only=True),
                        method="dagger-"+NAMES[idx],mode=[0,0,1] if idx==2 else [0,1,0],rule=NAMES[idx])
            torch.save(pack,path)
    locked=dict(name=NAMES[idx],weights_sha256=digest(path),dev_sha256=digest(DEV),
                selection=digest(OUT/"selection.json"),ranks=dict(zip(NAMES,sel["rank"])))
    if (OUT/"locked-before-dev.json").exists():assert read(OUT/"locked-before-dev.json")==locked
    else:save(OUT/"locked-before-dev.json",locked)
    print("frozen",locked,flush=True)
    bank=rows(DEV);cache=CACHE/"dev-dagger";cache.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4,initializer=init_dev) as pool:
        jobs=[pool.submit(journey,r) for r in bank if not (cache/(r["name"]+".json")).exists()]
        for i,f in enumerate(as_completed(jobs),1):
            ep=f.result();save(cache/(ep["map"]+".json"),ep)
            if i%25==0:print("dagger dev",i,flush=True)
    episodes=[read(cache/(r["name"]+".json")) for r in bank]
    save(path.with_suffix(".physical.json"),dict(weights_sha256=digest(path),dev_sha256=digest(DEV),
         summary=summarize(episodes),episodes=episodes))
    print(summarize(episodes),flush=True)


if __name__=="__main__":main()
