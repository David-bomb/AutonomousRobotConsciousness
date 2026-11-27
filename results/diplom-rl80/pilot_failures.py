"""Post-mortem of a frozen pilot on fresh training geometries. Full-map queries follow action collection."""
import argparse
import hashlib
from collections import Counter
from importlib import import_module
import json
from pathlib import Path
import sys


def diagnose(root, checkpoint, output):
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from maze.environment import Environment, Status
    from scripts.audit_automaton import Reachability
    data = import_module("results.E-0016.data")
    policy = import_module("results.E-0016.policy")
    torch.set_num_threads(1)
    net = policy.Policy().eval()
    net.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    rows = data.rows(data.OUT/"data/train.jsonl")[200:300]
    results = []
    with torch.inference_mode():
        for row in rows:
            env = Environment(data.level(row))
            seen, history, unchanged = set(), [], {}
            repeated_exact = False
            while env.status == Status.RUNNING:
                view = env.view
                if view in seen:
                    repeated_exact = True
                    break
                seen.add(view)
                point_key = (view.position,view.memory)
                back_to_known = point_key in unchanged
                gap = env.steps-unchanged[point_key] if back_to_known else None
                unchanged[point_key] = env.steps
                x, valid = policy.features(view)
                scores = net(torch.from_numpy(x[None]),torch.from_numpy(valid[None]))[0]
                move = policy.MOVES[scores.argmax().item()]
                history.append((env.position,env.fuel,env.health,env.refuels,env.repairs,back_to_known,gap,move))
                env.step(move)
            end = len(history)
            history.append((env.position,env.fuel,env.health,env.refuels,env.repairs,False,None,None))
            out = dict(geometry=row["geometry"],tier=row["tier"],outcome="exact_cycle" if repeated_exact
                       else env.status.name,steps=end)
            if not repeated_exact and env.status not in (Status.SUCCESS, Status.LIMIT):
                oracle=Reachability(env.level)
                lo,hi=0,end
                while hi-lo>1:
                    m=(lo+hi)//2
                    if oracle.possible(*history[m][:3],env.limit-m):
                        lo=m
                    else:
                        hi=m
                before=history[lo]
                assert oracle.possible(*before[:3],env.limit-lo)
                assert not oracle.possible(*history[hi][:3],env.limit-hi)
                out.update(loss=hi, before_first=not(before[3] or before[4]),
                           after_both=bool(before[3] and before[4]), revisited=before[5],
                           revisit_age=before[6],
                           prior_revisits=sum(item[5] for item in history[max(0,lo-8):lo+1]),
                           degree=len(oracle.edges[before[0]]))
            results.append(out)
    losing=[r for r in results if "loss" in r]
    result=dict(checkpoint=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                scope="train[200:300] of E-0016, excluded from pilot200; external post-mortem",
                outcomes=Counter(r["outcome"] for r in results), summary={
                "resource_failures":len(losing),"before_first":sum(r["before_first"] for r in losing),
                "after_both":sum(r["after_both"] for r in losing),
                "revisit_before_loss":sum(r["revisited"] for r in losing),
                "any_revisit_in_last_nine":sum(r["prior_revisits"]>0 for r in losing),
                "median_first_loss":float(np.median([r["loss"] for r in losing])) if losing else None},
                rows=results)
    encoded=json.dumps(result,ensure_ascii=False,indent=2).encode()
    if output.exists() and output.read_bytes()!=encoded:
        raise ValueError("Refusing to replace a different audit")
    if not output.exists():
        output.write_bytes(encoded)
    print(json.dumps([result["outcomes"],result["summary"]]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("project","checkpoint","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    diagnose(args.project.resolve(), args.checkpoint, args.output)
