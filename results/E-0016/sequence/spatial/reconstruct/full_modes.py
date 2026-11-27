"""Дополнительный опубликованный контроль полных коэффициентов, без выбора по dev."""
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from ....data import digest, read, rows, save
from ....dev import evaluate
from ..rooms.physical import summarize
from .adapter import load
from .bank import RUN, CACHE, DEV


def init(files):
    global controllers
    controllers={k:load(p.read_bytes()) for k,p in files.items()}


def trip(job):
    tag,row=job
    return tag,evaluate(controllers[tag],[row])[1][0]


def main():
    lock=read(RUN/"locked-before-dev.json")
    tags={"marginal1":[0,1,0],"direct1":[0,0,1]}
    files={}
    for tag,mode in tags.items():
        path=RUN/(tag+".pt")
        if not path.exists():
            data=torch.load(RUN/"rl.pt",weights_only=True)
            data.update(method=tag,mode=mode)
            torch.save(data,path)
        files[tag]=path
    spec=dict(hashes={k:digest(p) for k,p in files.items()},initial_lock=digest(RUN/"locked-before-dev.json"),
              dev=digest(DEV),selection_eligible=False)
    if (RUN/"pure-lock.json").exists():assert read(RUN/"pure-lock.json")==spec
    else:save(RUN/"pure-lock.json",spec)
    bank=rows(DEV)
    for k in tags:(CACHE/("dev-"+k)).mkdir(exist_ok=True)
    jobs=[(k,r) for r in bank for k in tags if not (CACHE/("dev-"+k)/(r["name"]+".json")).exists()]
    with ProcessPoolExecutor(3,initializer=init,initargs=(files,)) as pool:
        fs=[pool.submit(trip,job) for job in jobs]
        for i,f in enumerate(as_completed(fs),1):
            k,ep=f.result();save(CACHE/("dev-"+k)/(ep["map"]+".json"),ep)
            if i%50==0:print("pure",i,flush=True)
    for k,path in files.items():
        episodes=[read(CACHE/("dev-"+k)/(r["name"]+".json")) for r in bank]
        save(path.with_suffix(".physical.json"),dict(weights_sha256=spec["hashes"][k],dev_sha256=spec["dev"],
              summary=summarize(episodes),episodes=episodes))
        print(k,summarize(episodes),flush=True)


if __name__=="__main__":main()
