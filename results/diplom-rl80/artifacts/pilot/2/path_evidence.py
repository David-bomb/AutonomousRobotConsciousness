"""Train-only probe of whether exact-distance path evidence discriminates viable first moves."""
import argparse
from collections import Counter
from importlib import import_module
import hashlib
import json
from pathlib import Path
import sys


def run(project, checkpoint, labels):
    sys.path.insert(0, str(project))
    import torch
    from maze.bridge_automaton import bridge_evidence
    from maze.environment import Environment
    from maze.grid import locations
    from maze.resource_automaton import Mode, Service
    data = import_module("results.E-0016.data")
    policy = import_module("results.E-0016.policy")
    torch.set_num_threads(1)
    net = policy.Policy().eval()
    net.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
    reference = {r["geometry"]:set(r["can_still_finish"]) for r in json.loads(labels.read_bytes())["records"]}
    # The pilot was trained on the first 200 geometries, and before this audit the formula was fixed.
    rows = data.rows(data.OUT/"data/train.jsonl")[200:]
    stats = Counter()
    with torch.inference_mode():
        for r in rows:
            if r["tier"] != "H4":
                continue
            v = Environment(data.level(r)).view
            x, legal = policy.features(v)
            logits = net(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))[0].tolist()
            base = max(range(8), key=logits.__getitem__)
            stock, read, symbol, mode = min([(v.fuel,v.fuel_distance,"F",Mode.FUEL),
                                              (v.health,v.repair_distance,"R",Mode.REPAIR)],key=lambda row:row[0])
            s = Service(mode,stock,read,tuple(locations(v.memory,symbol)))
            evidence = bridge_evidence(v,s,None,True) or bridge_evidence(v,s,None)
            winner = max((i for i in range(8) if legal[i]),
                         key=lambda i:(evidence.get(policy.MOVES[i],0),logits[i]))
            correct = reference[r["geometry"]]
            a = policy.MOVES[base].name in correct
            b = policy.MOVES[winner].name in correct
            stats["maps"]+=1
            stats["base_viable"]+=a
            stats["evidence_viable"]+=b
            stats["gained"]+=b and not a
            stats["lost"]+=a and not b
    return dict(split="H4 train geometries 201..800, full start, no hint",
                native_computation="bridge_evidence using ordinary Observation only; studied historic prior",
                weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),outcomes=stats)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("project","checkpoint","labels","output"):
        ap.add_argument(arg,type=Path)
    a=ap.parse_args()
    result=run(a.project.resolve(),a.checkpoint,a.labels)
    body=json.dumps(result,ensure_ascii=False,indent=2).encode()
    if a.output.exists() and a.output.read_bytes()!=body:
        raise ValueError("Previous audit differs")
    a.output.write_bytes(body)
    print(json.dumps(result["outcomes"]))
