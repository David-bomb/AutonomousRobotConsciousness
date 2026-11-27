"""Одна новая тренировочная сотня, исключённая из выбора всех коэффициентов комнаты."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import re
import numpy as np
import torch
from maze.datasets import geometry
from maze.environment import Environment,Status
from ....data import OUT,level,generate,rows,read,save,digest
from ....policy import MOVES
from ....teacher import Teacher
from ..fresh import fingerprints
from ..model import image
from .cells import HERE,representation
from .physical import ROOT,summarize
from .model import RoomPolicy


def register():
    manifest=HERE/"independent-bank.json"
    if manifest.exists():
        assert digest(HERE/"independent.jsonl")==read(manifest)["sha256"]
        return
    source=read(HERE.parent/"bank-audit.json")
    keys=fingerprints()
    paths=[OUT/r["path"] for r in source["banks"]]
    paths.extend(HERE.parent/(n+".jsonl") for n in read(HERE.parent/"decisions-bank.json")["groups"])
    paths.append(HERE/"selection.jsonl")
    field=re.compile(rb'"geometry"\s*:\s*"([0-9a-f]{64})"')
    for path in paths:
        with path.open("rb") as f:
            for line in f:keys.add(field.search(line).group(1).decode())
    assert len(keys)==read(HERE/"bank.json")["excluded"]+200
    before=len(keys);seed=9164502
    bank=generate(("room-independent",seed));dest=HERE/"independent.jsonl"
    with dest.open("x",encoding="utf8",newline="\n") as f:
        for r in bank:
            key=geometry(r["grid"]);assert key==r["geometry"] and key not in keys
            keys.add(key);f.write(json.dumps(r,separators=(",",":"))+"\n")
    assert len(keys)==before+100
    save(manifest,dict(seed=seed,count=100,excluded=before,duplicates=0,sha256=digest(dest),
         use="independent full journeys and first irreversible choice; withheld from training and selection"))


def init(path):
    global net,w
    torch.set_num_threads(1)
    pack=torch.load(path,weights_only=True,map_location="cpu")
    net=RoomPolicy(pack);w=pack["linear"].numpy()


def trip(row):
    e=Environment(level(row));t=Teacher(e.level);cache={}
    tests={};runs={}
    with torch.inference_mode():
        for mode in ("base","room"):
            e.reset();lost=False
            while e.status==Status.RUNNING:
                v=e.view
                if v not in cache:
                    z,ok=net.directions(v)
                    cache[v]=(int(np.where(ok,z[:,0],-1e12).argmax()),int(np.where(ok,z@w,-1e12).argmax()),int(ok.sum()))
                a,b,degree=cache[v]
                if mode=="base" and not lost and not (e.refuels and e.repairs) and e.steps<150:
                    _,safe=t.labels(e)
                    slack=min(v.fuel-v.fuel_distance,v.health-v.repair_distance)
                    names=[]
                    if degree>1 and slack<=2 and "first_tight" not in tests:names.append("first_tight")
                    if not safe[a] and safe.any():lost=True;names.append("first_loss")
                    if names:
                        board,m=image(v)
                        _,stats,added,blocked=representation(board,m,len(v.memory[0]),len(v.memory),net.counts)
                        family=net.family(torch.from_numpy(stats[None])).softmax(-1)[0].numpy()
                        real=np.asarray([list(s) for s in row["grid"]])!="#"
                        sample=dict(step=e.steps+1,slack=slack,previous_safe=bool(safe[a]),candidate_safe=bool(safe[b]),
                            changed=a!=b,seen_services=int(bool(e.refuels))+int(bool(e.repairs)),family_pred=int(family.argmax()),
                            room_probability=float(family[3]),inferred=int(added.sum()),actually_road=int((added&real).sum()),
                            walls=int(blocked.sum()),actually_wall=int((blocked&~real).sum()))
                        for n in names:tests[n]=sample
                e.step(MOVES[a if mode=="base" else b])
            runs[mode]=dict(map=row["name"],tier=row["tier"],outcome=e.status.name,steps=e.steps,refuels=e.refuels,repairs=e.repairs)
    t.remaining.cache_clear()
    return dict(map=row["name"],tier=row["tier"],family=row["family"],tests=tests,runs=runs)


def main(reg_only):
    register()
    if reg_only:return
    out=ROOT/"runs/rooms3000";path=out/"rl.pt"
    assert read(out/"locked-before-dev.json")["weights_sha256"]==digest(path)
    src=HERE/"independent.jsonl";bank=rows(src)
    cache=ROOT/"cache/rooms-independent";cache.mkdir(exist_ok=True)
    cfg=dict(weights=digest(path),data=digest(src),source=digest(HERE/"independent.py"))
    if (cache/"config.json").exists():assert read(cache/"config.json")==cfg
    else:save(cache/"config.json",cfg)
    with ProcessPoolExecutor(2,initializer=init,initargs=(path,)) as pool:
        for j,r in enumerate(pool.map(trip,[r for r in bank if not (cache/(r["name"]+".json")).exists()]),1):
            save(cache/(r["map"]+".json"),r)
            if j%20==0:print("independent",j,flush=True)
    results=[read(cache/(r["name"]+".json")) for r in bank]
    s={k:summarize([r["runs"][k] for r in results]) for k in ("base","room")}
    h4={}
    for family in ("corridors","branching","loops","rooms"):
        use=[r for r in results if r["tier"]=="H4" and r["family"]==family]
        h4[family]=dict(count=len(use),base=summarize([r["runs"]["base"] for r in use]),room=summarize([r["runs"]["room"] for r in use]))
    probes={}
    for k in ("first_tight","first_loss"):
        items=[r["tests"][k] for r in results if k in r["tests"]]
        probes[k]=dict(count=len(items),slack01=sum(v["slack"] in (0,1) for v in items),
                changed=sum(v["changed"] for v in items),base_safe=sum(v["previous_safe"] for v in items),
                room_safe=sum(v["candidate_safe"] for v in items))
    save(out/"independent.json",dict(config=cfg,summary=s,h4=h4,probes=probes,episodes=results))
    print(s,probes,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--register",action="store_true")
    main(ap.parse_args().register)
