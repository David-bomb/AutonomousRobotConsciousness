"""Train last-departure decisions against terminal success on physical training episodes."""
from copy import copy
from hashlib import sha256

import numpy as np
import torch
from departure_policy import Departure, load
from early_trials import physical
from exact_train import balanced
from renewable_train import EXP, HERE, Environment, Status, data, digest, moves, save

OUT = HERE / "departure-search"


def trials(tag, rows, population, net):
    folder = OUT / tag
    folder.mkdir(exist_ok=True)
    totals = np.zeros(len(population), int)
    records = []
    protocol = digest(OUT / "protocol.json")
    for i,row in enumerate(rows):
        dest = folder / (row["name"]+".json")
        if dest.exists():
            group = data.read(dest)
            assert group["protocol"] == protocol and group["population"] == population.tolist()
        else:
            start = Environment(data.level(row))
            past = (None,None)
            prefix = bytearray()
            seen = set()
            while start.status == Status.RUNNING and not net.route(start.view,net.cfg):
                v = start.view
                if (v,past) in seen:
                    break
                seen.add((v,past))
                action = int(net.early_scores(v,past).argmax())
                prefix.append(action)
                past = v.fuel_distance,v.repair_distance
                start.step(moves[action])
            gate = start.status == Status.RUNNING and net.route(start.view,net.cfg)
            predictions,episodes = {},[]
            for j in range(len(population)):
                env = copy(start)
                env._memory = copy(start._memory)
                old, seen, commands = past,set(),bytearray(prefix)
                while gate and env.status == Status.RUNNING:
                    v = env.view
                    key = v,old
                    if key in seen:
                        break
                    seen.add(key)
                    if key not in predictions:
                        q,x = net.late_parts(v,old)
                        predictions[key] = (q+population@x.T).argmax(1)
                    action = int(predictions[key][j])
                    commands.append(action)
                    old = v.fuel_distance,v.repair_distance
                    env.step(moves[action])
                outcome = env.status.name if env.status != Status.RUNNING else "exact_cycle"
                episodes.append(dict(map=row["name"],tier=row["tier"],model=j,outcome=outcome,gate=gate,
                                     steps=len(commands),actions_sha256=sha256(commands).hexdigest()))
            group = dict(protocol=protocol, population=population.tolist(), episodes=episodes)
            save(dest,group)
        records.extend(group["episodes"])
        totals += np.array([e["outcome"]=="SUCCESS" for e in group["episodes"]])
        if (i+1)%10 == 0:
            print(tag,i+1,totals.tolist(),flush=True)
    save(OUT/(tag+".json"),dict(wins=totals.tolist(),episodes=records,options=population.tolist()))
    return totals


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    parent = HERE / "protected-search/fixed/model.pt"
    assert digest(parent) == "92b9b74fbd924161e79bf37b7e7b3dadfbafcd0bc2cd7fb0bbdcaee3d18665b1"
    diagnosis = data.read(HERE/"late-errors/summary.json")["episodes"]
    banks = [HERE/n for n in ("return-search/bank.jsonl","handover-search/train.jsonl")]
    lookup = {r["name"]:r for bank in banks for r in data.rows(bank)}
    fit = [lookup[e["map"]] for e in diagnosis]
    pool = EXP/"sequence/spatial/large.jsonl"
    select = balanced(data.rows(pool)[5200:5300],25)
    assert not ({r["geometry"] for r in fit}&{r["geometry"] for r in select})
    eye = np.eye(12)
    population = np.vstack((np.zeros(12),12*eye[:6],eye[6:],5*eye[0]+9*eye[4]))
    pack = torch.load(parent,map_location="cpu",weights_only=True)
    pack.update(format="departure-1",departure_samples=36,departure=torch.zeros(12,dtype=torch.float64))
    save(OUT/"protocol.json",dict(parent=digest(parent),fit=[r["name"] for r in fit],select=[r["name"] for r in select],
        select_sha256=digest(pool),diagnosis=digest(HERE/"late-errors/summary.json"),options=population.tolist(),samples=36,
        sources={n:digest(HERE/n) for n in ("departure_train.py","departure_policy.py","departure_plan.cpp",
                "departure_worlds.cpp","departure.dll","proper_frontier.dll","morphology_worlds.cpp")},
        rule="Apply only after the fixed handoff when no known route to the exit or safely returnable new window exists. Rank complete paths through a common reachable frontier and their residual stocks across generated road hypotheses. Fit whole episodes on independently identified post-handoff losses and matched successes. Top three nonzero settings, their mean, and parent compete on a distinct balanced training hundred. Rank total success with parent first on ties. Freeze before physical dev and ablation."))
    net = Departure(pack)
    a = trials("fit",fit,population,net)
    lead = population[np.argsort(-a[1:],kind="stable")[:3]+1]
    finalists = np.vstack((population[0],lead,lead.mean(0)))
    b = trials("select",select,finalists,net)
    theta = finalists[int(b.argmax())]
    path = OUT/"model.pt"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        torch.save(dict(pack,departure=torch.tensor(theta,dtype=torch.float64)),tmp)
        tmp.replace(path)
    save(OUT/"frozen.json",dict(model_sha256=digest(path),coefficients=theta.tolist(),fit=a.tolist(),select=b.tolist(),
                                protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):
        physical(OUT,tag,load(path.read_bytes(),ablate))


if __name__ == "__main__":
    main()
