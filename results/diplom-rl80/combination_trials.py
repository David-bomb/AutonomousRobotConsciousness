"""Prospective, map-disjoint cross of independently frozen early and late policies."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import import_module
import shutil
import torch
from combination_policy import load
from early_trials import physical
from renewable_train import HERE, Environment, Status, data, digest, moves, save

OUT=HERE/"combination-search"
PLAN=HERE/"joint_selection_plan.json"
fork=import_module("results.E-0016.sequence.branches").fork


def freeze():
    plan=data.read(PLAN)
    sources=plan["early_families_in_order"]
    late_path=HERE/"departure-search/model.pt"
    assert digest(late_path)==data.read(late_path.with_name("frozen.json"))["model_sha256"]
    late=torch.load(late_path,weights_only=True,map_location="cpu")
    dirs=[];hashes={}
    for i,name in enumerate(sources):
        path=HERE/name/"model.pt"
        assert digest(path)==data.read(path.with_name("frozen.json"))["model_sha256"]
        p=torch.load(path,weights_only=True,map_location="cpu")
        for k in ("early","cautious_state"):
            assert all(torch.equal(v,late[k][j]) for j,v in p[k].items())
        for k in ("base","safety"):
            assert all(torch.equal(v,late["late"][k][j]) for j,v in p["late"][k].items())
        for k in ("linear","tables"):assert torch.equal(p["late"][k],late["late"][k])
        for j in range(2):
            code=f"e{i}l{j}"
            dest=OUT/code;dest.mkdir(exist_ok=True)
            model=dest/"model.pt"
            if not model.exists():
                pack=dict(p,format="combined-departure-1",early_format=p["format"],use_departure=bool(j),
                          departure_samples=late["departure_samples"],departure=late["departure"])
                temp=model.with_suffix(".tmp");torch.save(pack,temp);temp.replace(model)
            hashes[code]=dict(model_sha256=digest(model),early=name,early_sha256=digest(path),
                              late=plan["late_families_in_order"][j],departure_source_sha256=digest(late_path))
            dirs.append(code)
    files=("combination_trials.py","combination_policy.py","departure_policy.py","departure.dll","proper_frontier.dll",
           "lookahead_policy.py","sensor_lookahead.dll","exact_policy.py","exact_mission.dll","protected_policy.py",
           "tree_policy.py","tree_paths.dll","morph_policy.py","morphology_worlds.dll","origin_trace.py",
           "service_window_policy.py","renewable_policy.py")
    save(OUT/"lock.json",dict(policies=hashes,order=dirs,plan_sha256=digest(PLAN),banks_sha256=digest(OUT/"banks.json"),
                              implementation={n:digest(HERE/n) for n in files}))
    return dirs


def initialize():
    global combinations,locks
    torch.set_num_threads(1)
    lock=data.read(OUT/"lock.json");locks=digest(OUT/"lock.json")
    combinations={}
    for name in lock["order"]:
        path=OUT/name/"model.pt"
        assert digest(path)==lock["policies"][name]["model_sha256"]
        combinations[name]=load(path.read_bytes())


def finish(env,old,net,values,actions):
    trail=[];seen=set()
    while env.status==Status.RUNNING:
        v=env.view;key=v,old
        assert net.early.route(v,net.early.cfg)
        mark=key,env.limit-env.steps
        if mark in values:
            outcome,n,token=values[mark];break
        if key in seen:
            outcome,n,token="exact_cycle",0,sha256(b"exact_cycle").digest();break
        seen.add(key)
        if key not in actions:
            net.previous=old
            actions[key]=moves.index(net.choose(v))
        a=actions[key];trail.append((mark,a));old=v.fuel_distance,v.repair_distance
        env.step(moves[a])
    else:
        outcome,n=env.status.name,0
        token=sha256(outcome.encode()).digest()
    for mark,a in reversed(trail):
        n+=1;token=sha256(bytes([a])+token).digest();values[mark]=outcome,n,token
    return outcome,n,token.hex()


def score(row):
    dest=OUT/"rows"/(row["name"]+".json")
    if dest.exists():
        record=data.read(dest)
        assert record["lock"]==locks and record["geometry"]==row["geometry"]
        return record
    values=[{},{}];actions=[{},{}];measurements={}
    for i in range(4):
        controller=combinations[f"e{i}l0"].early
        env=Environment(data.level(row));past=(None,None);visited=set();prefix=bytearray()
        while env.status==Status.RUNNING and not controller.route(env.view,controller.cfg):
            v=env.view;k=v,past
            if k in visited:break
            visited.add(k)
            chosen=int(controller.early_scores(v,past).argmax())
            prefix.append(chosen);past=v.fuel_distance,v.repair_distance
            env.step(moves[chosen])
        gate=env.status==Status.RUNNING and controller.route(env.view,controller.cfg)
        for j in range(2):
            name=f"e{i}l{j}"
            if gate:
                # A zero return-selected departure head takes precisely the common joint400 branch.
                key=j if combinations[name].late is not None else 0
                outcome,n,token=finish(fork(env),past,combinations[name],values[key],actions[key])
            else:
                outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
                n=0;token=sha256(outcome.encode()).hexdigest()
            measurements[name]=dict(outcome=outcome,gate=gate,prefix=len(prefix),suffix=n,
                                    prefix_sha256=sha256(prefix).hexdigest(),suffix_chain_sha256=token)
    result=dict(map=row["name"],tier=row["tier"],geometry=row["geometry"],lock=locks,results=measurements)
    save(dest,result)
    return result


def main():
    torch.set_num_threads(1)
    dirs=freeze();(OUT/"rows").mkdir(exist_ok=True)
    groups=[data.rows(OUT/(str(s)+".jsonl")) for s in data.read(PLAN)["seeds"]]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as executor:
        for i,_ in enumerate(executor.map(score,[r for pair in zip(*groups,strict=True) for r in pair])):
            if (i+1)%25==0:print(i+1,"paired training maps completed",flush=True)
    record={}
    for code in dirs:
        counts=[];h4=0
        for group in groups:
            episodes=[data.read(OUT/"rows"/(r["name"]+".json")) for r in group]
            wins=[e for e in episodes if e["results"][code]["outcome"]=="SUCCESS"]
            counts.append(len(wins));h4+=sum(e["tier"]=="H4" for e in wins)
        record[code]=dict(counts=counts,h4=h4,ranking=[min(counts),sum(counts),h4,-dirs.index(code)])
    winner=max(dirs,key=lambda code:record[code]["ranking"])
    dest=OUT/"model.pt"
    if not dest.exists():
        tmp=dest.with_suffix(".tmp");shutil.copyfile(OUT/winner/"model.pt",tmp);tmp.replace(dest)
    assert digest(dest)==digest(OUT/winner/"model.pt")
    save(OUT/"frozen.json",dict(winner=winner,record=record,model_sha256=digest(dest),lock_sha256=digest(OUT/"lock.json")))
    # Verify eight ordinary full rollouts independently of the shared training suffixes.
    checks=[]
    for group in groups:
        for tier in ("H1","H2","H3","H4"):
            row=next(r for r in group if r["tier"]==tier);env=Environment(data.level(row))
            policy=load(dest.read_bytes());cache={}
            while env.status==Status.RUNNING:env.step(policy.choose(env.view,cache))
            expected=data.read(OUT/"rows"/(row["name"]+".json"))["results"][winner]
            assert (env.status==Status.SUCCESS)==(expected["outcome"]=="SUCCESS")
            if expected["outcome"]!="exact_cycle":
                assert env.status.name==expected["outcome"] and env.steps==expected["prefix"]+expected["suffix"]
            checks.append(dict(map=row["name"],outcome=env.status.name,steps=env.steps))
    save(OUT/"replay.json",dict(checks=checks,model_sha256=digest(dest)))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(dest.read_bytes(),ablate))


if __name__=="__main__":main()
