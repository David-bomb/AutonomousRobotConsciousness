"""Проверки фактических измерений, восстановления и вершины абляции на полных поездках."""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import Counter
from copy import copy
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from .adapter import MOVES,memory_fields
from .common import DEV,HERE,digest,group,level,read,rows,write
from .physical import aggregate,episode


def controller_for(tag):
    path=HERE/"weights"/(tag+".pt")
    frozen=read(path.with_suffix(".json"))
    if frozen.get("patch_adapter_sha256"):
        from .patch_adapter import Controller
    elif frozen.get("route_adapter_sha256"):
        from .route_adapter import Controller
    else:
        from .adapter import Controller
    return Controller(path,digest(path))


def check_bounds(tag):
    controller=controller_for(tag)
    total=changed=reverse=0
    curriculum=group("probe")[:12]
    rng=np.random.default_rng(161128)
    for row in curriculum:
        env=Environment(level(row))
        controller.reset()
        cache={}
        for step in range(48):
            v=env.view
            legal=np.array([destination(v.window,(1,1),m) is not None for m in MOVES])
            previous=controller.previous
            fields,bounds=memory_fields(v,previous,legal)
            missing,other=memory_fields(v,(None,None),legal)
            changed+=int((bounds!=other).any())
            greedy=controller.choose(v,cache)
            # Тот же запрос должен обновить оба числа и при найденном в кеше ходе.
            controller.previous=previous
            assert greedy==controller.choose(v,cache)
            assert controller.previous==(v.fuel_distance,v.repair_distance)
            for a in np.flatnonzero(legal):
                fork=copy(env)
                fork._memory=copy(env._memory)
                after=fork.step(MOVES[a])
                labels=[after.fuel_distance-v.fuel_distance+1,after.repair_distance-v.repair_distance+1]
                if v.health<v.fuel: labels.reverse()
                assert bounds[a,0,labels[0]] and bounds[a,1,labels[1]]
                assert other[a,0,labels[0]] and other[a,1,labels[1]]
                total+=2
                if fields[a,0,6]>.5:
                    assert bounds[a].sum()==2
                    reverse+=1
            assert env.view==v
            move=MOVES[int(rng.choice(np.flatnonzero(legal)))] if step%5==2 else greedy
            env.step(move)
            if env.status!=Status.RUNNING: break
        controller.reset()
        assert controller.previous==(None,None)
    return dict(executed_labels=total,states_with_tighter_bounds=changed,known_reverse_edges=reverse)


def replay(data,bank):
    all_unique=Counter();first_events=Counter();feasible_stock=Counter()
    for e in data["episodes"]:
        env=Environment(level(bank[e["map"]]))
        previous=(None,None); old_kind=None
        seen=set();first=True
        for action in e["actions"]:
            v=env.view;cur=v.fuel_distance,v.repair_distance
            slack=[v.fuel-cur[0],v.health-cur[1]]
            kind=min(range(2),key=slack.__getitem__)
            tight=slack[kind]<=3 and cur[kind]>0
            failed_probe=tight and old_kind==kind and previous[kind] is not None and previous[kind]<cur[kind]
            state=v,previous,old_kind
            chosen=MOVES[int(action)]
            reversed_move=v.last_move is not None and chosen.value==tuple(-d for d in v.last_move.value)
            env.step(chosen)
            assert env.moved
            down=(env.view.fuel_distance,env.view.repair_distance)[kind]<cur[kind]
            if state not in seen:
                seen.add(state)
                if tight: all_unique.update(tight=1,tight_down=int(down))
                if failed_probe:
                    all_unique.update(after_probe=1,after_probe_down=int(down),after_probe_reverse=int(reversed_move))
                    if slack[kind]>=0:
                        feasible_stock.update(exposures=1,down=int(down),reverse=int(reversed_move))
            if first and failed_probe:
                first_events.update(maps=1,down=int(down),reverse=int(reversed_move))
                first=False
            old_kind=kind if tight else None
            previous=cur
        assert (env.status.name,env.steps,env.refuels,env.repairs)==(e["outcome"],e["steps"],e["refuels"],e["repairs"])
    return dict(unique_within_map=dict(all_unique),first_failed_probe_by_map=dict(first_events),
                unique_nonnegative_slack_after_probe=dict(feasible_stock))


def compare(a,b):
    gains=[];loss=[];both=Counter()
    for r,s in zip(a,b,strict=True):
        assert r["map"]==s["map"]
        old=r["outcome"]=="SUCCESS";new=s["outcome"]=="SUCCESS"
        if old!=new: (gains if new else loss).append(s["map"])
        if min(r["refuels"],r["repairs"],s["refuels"],s["repairs"])>0:
            both.update(maps=1,old=int(old),new=int(new))
    return dict(gains=gains,losses=loss,both_each=dict(both))


def audit(tag):
    torch.set_num_threads(1)
    bank={r["name"]:r for r in rows(DEV)}
    ref=read(HERE/"dev/mission600-baseline.json")
    mem=read(HERE/"dev"/(tag+"-honest.json"))
    abl=read(HERE/"dev"/(tag+"-ablated.json"))
    path=HERE/"weights"/(tag+".pt");sha=digest(path)
    assert sha==mem["weights_sha256"]==abl["weights_sha256"]
    assert all(r["dev_sha256"]==digest(DEV) for r in (ref,mem,abl))
    for r in (ref,mem,abl):
        assert aggregate(r["episodes"])==r["summary"]
    results={k:replay(data,bank) for k,data in (("baseline",ref),("honest",mem),("ablated",abl))}
    bounds=check_bounds(tag)
    # Пошаговое воспроизведение без кеша на первой, последней и двух разных
    # чувствительных к памяти картах, включая успех и проигрыш, если есть оба.
    different=[i for i,(r,s) in enumerate(zip(mem["episodes"],abl["episodes"])) if r["actions"]!=s["actions"]]
    picks={0,99}
    for success in (False,True):
        options=[i for i in different if (mem["episodes"][i]["outcome"]=="SUCCESS")==success]
        if options: picks.add(min(options,key=lambda i:mem["episodes"][i]["steps"]))
    ctl=controller_for(tag)
    for i in sorted(picks):
        e=mem["episodes"][i]
        assert episode(bank[e["map"]],ctl,"honest",False)==e
    write(HERE/("audit-"+tag+".json"),dict(tag=tag,sha256=sha,dev_sha256=digest(DEV),physical_replays=300,
          uncached_maps=[mem["episodes"][i]["map"] for i in sorted(picks)],bounds=bounds,recovery=results,
          versus_baseline=compare(ref["episodes"],mem["episodes"]),
          versus_ablation=compare(abl["episodes"],mem["episodes"])))
    print("audit",tag,bounds,results,flush=True)


if __name__=="__main__":audit(sys.argv[1])
