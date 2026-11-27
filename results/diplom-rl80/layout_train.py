"""Fit a single observable landmark split to recorded paired training rollouts."""
from collections import Counter
from hashlib import sha256
import torch
from layout_router import layout_features, load, select, sites
from renewable_train import EXP, HERE, Environment, Status, data, digest, moves, save

OUT = HERE / "layout-search"
FIXED = HERE / "service-window/fixed"
PROPER = HERE / "proper-search"


def features(v):
    x,y = v.position
    return layout_features(len(v.memory[0]),len(v.memory),(x+v.dx,y+v.dy),*sites(v.memory))


def outcomes(filename, setting, idx):
    obj = data.read(filename)
    j = obj["configs"].index(setting)
    return {e["map"]:e["outcome"] == "SUCCESS" for e in obj["episodes"] if e[idx] == j}


def paired(part, bank):
    a = outcomes(HERE/"service-window/learned"/(part+".json"), [9,9,-101], "model")
    b = outcomes(PROPER/(part+".json"), [1,4,-101], "policy")
    return [(n,features(Environment(data.level(row)).view),(a[n],b[n]))
            for n,row in bank.items() if n in a and n in b]


def score(rule,rows):
    return sum(outcomes[select(rule,x)] for _,x,outcomes in rows)


def fit():
    bank_file = HERE/"return-search/bank.jsonl"
    bank = {r["name"]:r for r in data.rows(bank_file)}
    fixed_weights = FIXED/"model.pt"
    proper_weights = PROPER/"model.pt"
    save(OUT/"protocol.json", dict(bank_sha256=digest(bank_file),parents=[digest(p) for p in (fixed_weights,proper_weights)],
        source_sha256={p:digest(HERE/p) for p in ("layout_router.py","layout_train.py")},
        fitting="Six fixed features available on every observation: minimum and maximum dimensions, distances from B to stations and distances between F-R. Fit at most one axis split on overlapping parent train fits, >=10 cases per leaf. Best two distinct training partitions join both constants on distinct selection maps. Accept split only if strictly above both constants on selection. Stable order breaks ties.",
        test="Frozen deployment reload, full dev and identical-map ablation. Verify exact parent outcome, action digest and duration on each map."))
    fit_cases, select_cases = [paired(part,bank) for part in ("fit","select")]
    assert set(e[0] for e in fit_cases).isdisjoint(e[0] for e in select_cases)
    constants = [dict(feature=-1,threshold=0,left=i,right=i) for i in (0,1)]
    options = []
    seen = set()
    for j in range(6):
        for threshold in sorted({x[j] for _,x,_ in fit_cases}):
            left = tuple(x[j]<=threshold for _,x,_ in fit_cases)
            if min(sum(left),len(left)-sum(left)) < 10:continue
            for i in (0,1):
                rule = dict(feature=j,threshold=threshold,left=i,right=1-i)
                assigned = tuple(select(rule,x) for _,x,_ in fit_cases)
                if assigned not in seen:
                    seen.add(assigned)
                    options.append(rule)
    shortlist = constants+sorted(options,key=lambda r:-score(r,fit_cases))[:2]
    results = [score(r,select_cases) for r in shortlist]
    rule = shortlist[results.index(max(results))]
    p = OUT/"model.pt"
    if not p.exists():
        pack = torch.load(fixed_weights,weights_only=True,map_location="cpu")
        proper = torch.load(proper_weights,weights_only=True,map_location="cpu")
        torch.save(dict(pack,format="layout-conditional-1",proper=proper["configuration"],rule=rule),p)
    save(OUT/"frozen.json",dict(rule=rule,model_sha256=digest(p),fits=[score(r,fit_cases) for r in shortlist],
        selection=results,shortlist=shortlist,paired_fit=[r[0] for r in fit_cases],paired_select=[r[0] for r in select_cases],
        fit_discordance=sum(a!=b for _,_,(a,b) in fit_cases),select_discordance=sum(a!=b for _,_,(a,b) in select_cases)))
    print("Frozen layout rule",rule,results,flush=True)
    return p,rule


def replay(p,rule,part,ablate):
    net = load(p.read_bytes(),ablate)
    parents = [{e["map"]:e for e in data.read(b/(part+".json"))["episodes"]} for b in (FIXED,PROPER)]
    episodes = []
    folder = OUT/part
    folder.mkdir(exist_ok=True)
    for r in data.rows(EXP/"data/dev.jsonl"):
        name = r["name"]
        env = Environment(data.level(r))
        parent = select(rule,features(env.view))
        ref = parents[parent][name]
        dest = folder/(name+".json")
        if dest.exists():
            e = data.read(dest)
        else:
            actions=bytearray(); cache={}
            while env.status == Status.RUNNING:
                m = net.choose(env.view,cache)
                actions.append(moves.index(m));env.step(m)
            e=dict(map=name,tier=r["tier"],head=parent,outcome=env.status.name,steps=env.steps,
                   refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest())
            save(dest,e)
        assert e["head"]==parent and all(e[k]==ref[k] for k in ("outcome","steps","refuels","repairs","actions_sha256"))
        episodes.append(e)
    counts=Counter(e["outcome"] for e in episodes)
    save(OUT/(part+".json"),dict(episodes=episodes,wins=[counts["SUCCESS"]],outcomes=dict(counts),weights_sha256=digest(p),
                                dev_sha256=digest(EXP/"data/dev.jsonl"),ablate=ablate))
    print(part,dict(counts),flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    p,rule=fit()
    for part,ablate in (("dev",False),("ablated",True)):
        replay(p,rule,part,ablate)
