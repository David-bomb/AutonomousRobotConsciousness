from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import torch
from ....data import OUT,rows,read,save,digest
from ....dev import evaluate
from ...belief.population import subset
from .. import transfer
from .cells import HERE
from .model import RoomPolicy,load

ROOT=HERE.parent.parent


def package():
    pack=torch.load(HERE/"supervised.pt",weights_only=True,map_location="cpu")
    pack["foundation"]=torch.load(ROOT/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu")
    return pack


def init_train():
    torch.set_num_threads(1)
    transfer.net=RoomPolicy(package())


def init_dev(path):
    global controller
    controller=load(path.read_bytes())


def one(row):
    # Та же функция официального dev физически проходит один эпизод.
    _,items=evaluate(controller,[row])
    return items[0]


def summarize(items):
    tiers=Counter(r["tier"] for r in items if r["outcome"]=="SUCCESS")
    return dict(wins=sum(tiers.values()),by_tier={t:tiers[t] for t in ("H1","H2","H3","H4")},
        outcomes=dict(Counter(r["outcome"] for r in items)),fuel=sum(bool(r["refuels"]) for r in items),
        repair=sum(bool(r["repairs"]) for r in items),either=sum(bool(r["refuels"] or r["repairs"]) for r in items),
        both=sum(bool(r["refuels"] and r["repairs"]) for r in items))


def main():
    out=ROOT/"runs/rooms3000";out.mkdir(exist_ok=True)
    cache=ROOT/"cache/rooms-physical";cache.mkdir(exist_ok=True)
    zero=[1]+[0]*9
    pre=[1,0,0,0,0,0,0,4,4,0]
    choices=[zero,pre]
    for scale in (4,12,25):
        for mix in ((1,1,0,0,0,0,0,0,0),(0,0,0,1,1,1,0,0,0),
                    (0,0,0,0,0,1,1,1,0),(.5,1,.5,0,.5,1,0,0,0),
                    (0,0,0,0,0,2,0,1,.5)):
            choices.append([1,*[v*scale for v in mix]])
    cfg=dict(source=digest(HERE/"physical.py"),model=digest(HERE/"model.py"),cells=digest(HERE/"cells.py"),
        held=digest(HERE/"held.json"),selection_bank=digest(HERE/"bank.json"),choices=choices,
        selection_rule="physical successes, then weakest tier; best distinct room candidate evaluated even if baseline wins")
    if (out/"config.json").exists():assert read(out/"config.json")==cfg
    else:save(out/"config.json",cfg)
    with ProcessPoolExecutor(3,initializer=init_train) as pool:
        screen=transfer.stage(pool,subset(rows(HERE.parent/"large.jsonl")[5500:5600],10),choices,cache/"screen")
        save(out/"screen.json",screen)
        ranking=sorted(screen["rank"],key=lambda r:(-r["wins"],-r["minimum"]))
        tests=[r["weights"] for r in ranking if r["weights"]!=zero][:4]
        if pre not in tests:tests.append(pre)
        tests.append(zero)
        sel=transfer.stage(pool,rows(HERE/"selection.jsonl"),tests,cache/"selection")
        save(out/"selection.json",sel)
    base=tests.index(zero)
    alternatives=[]
    for i,item in enumerate(sel["rank"]):
        changes=sum(r["episodes"][i]!=r["episodes"][base] for r in sel["records"])
        if i!=base and changes:alternatives.append(dict(**item,changed_train_episodes=changes))
    if not alternatives:raise ValueError("Ни один вариант не изменил учебную поездку")
    pick=max(alternatives,key=lambda r:(r["wins"],r["minimum"]))
    best=max(sel["rank"],key=lambda r:(r["wins"],r["minimum"]))
    dev=OUT/"data/dev.jsonl";path=out/"rl.pt"
    if not path.exists():
        pack=package();pack["linear"]=torch.tensor(pick["weights"],dtype=torch.float64)
        torch.save(pack,path)
    sig=digest(path)
    lock=dict(weights_sha256=sig,room_candidate=pick,best_including_baseline=best,
              pre_RL=sel["rank"][tests.index(pre)],baseline=sel["rank"][base],dev_sha256=digest(dev))
    if (out/"locked-before-dev.json").exists():assert read(out/"locked-before-dev.json")==lock
    else:save(out/"locked-before-dev.json",lock)
    bank=rows(dev)
    physical=cache/"dev";physical.mkdir(exist_ok=True)
    start=time.monotonic()
    with ProcessPoolExecutor(3,initializer=init_dev,initargs=(path,)) as pool:
        jobs=[pool.submit(one,r) for r in bank if not (physical/(r["name"]+".json")).exists()]
        for i,f in enumerate(as_completed(jobs),1):
            result=f.result();save(physical/(result["map"]+".json"),result)
            if i%20==0:print("physical dev",i,len(jobs),flush=True)
    episodes=[read(physical/(r["name"]+".json")) for r in bank]
    summary=summarize(episodes)
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(dev),summary=summary,
         episodes=episodes,seconds=time.monotonic()-start))
    profile={t:{fam:summarize([ep for ep,r in zip(episodes,bank) if r["family"]==fam and r["tier"]==t])
              for fam in ("corridors","branching","loops","rooms")} for t in ("H1","H2","H3","H4")}
    save(out/"dev-families.json",dict(weights_sha256=sig,groups=profile))
    print(lock,summary,flush=True)


if __name__=="__main__":main()
