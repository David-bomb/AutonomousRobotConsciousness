"""Pack the selected learned parameters together and replay every dev episode in the simulator."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import torch
from frontier_adapter import load

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.environment import Environment,Status
from maze.interface import Move
data=import_module("results.E-0016.data")
digest=lambda p:sha256(p.read_bytes()).hexdigest()


def main():
    d=HERE/"frontier-search"
    exp=PROJECT/"results/E-0016"
    decision=data.read(d/"frozen.json")
    protocol=data.read(d/"protocol.json")
    base_path=exp/"sequence/runs/mission600/s8.pt"
    assert decision["protocol_sha256"]==digest(d/"protocol.json") and protocol["base_sha256"]==digest(base_path)
    expected=data.read(d/"dev.physical.json")
    assert expected["coefficients"]==[decision["coefficients"]]
    frozen=d/"model.pt"
    if not frozen.exists():
        tmp=d/"model.pt.tmp"
        torch.save(dict(format="known-frontier-1",base=torch.load(base_path,map_location="cpu",weights_only=True),
                        weights=torch.tensor(decision["coefficients"],dtype=torch.float64)),tmp)
        tmp.replace(frozen)
    model=load(frozen.read_bytes())
    bank=exp/"data/dev.jsonl"
    cases=data.rows(bank)
    assert len(expected["episodes"])==len(cases)==100
    outcomes=Counter()
    tiers=Counter()
    rows=[]
    for case,reference in zip(cases,expected["episodes"]):
        env=Environment(data.level(case))
        actions=bytearray()
        cache={}
        while env.status==Status.RUNNING:
            if env.view not in cache:cache[env.view]=model.choose(env.view)
            m=cache[env.view]
            actions.append(tuple(Move).index(m))
            env.step(m)
        assert (reference["map"],reference["tier"],reference["outcome"],reference["steps"],reference["refuels"],reference["repairs"])==(
            case["name"],case["tier"],env.status.name,env.steps,env.refuels,env.repairs), case["name"]
        outcomes[env.status.name]+=1
        tiers[case["tier"]]+=env.status==Status.SUCCESS
        rows.append(dict(map=case["name"],geometry=case["geometry"],tier=case["tier"],outcome=env.status.name,
                         steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest()))
    record=dict(model_sha256=digest(frozen),adapter_sha256=digest(HERE/"frontier_adapter.py"),dev_sha256=digest(bank),
                decision_sha256=digest(d/"frozen.json"),outcomes=outcomes,tiers=tiers,steps=sum(r["steps"] for r in rows),
                matched_all_fields=len(rows),episodes=rows)
    out=d/"packed-dev.json"
    if out.exists():assert data.read(out)==record
    else:data.save(out,record)
    print(digest(frozen),dict(tiers),dict(outcomes),record["steps"])


if __name__=="__main__":main()
