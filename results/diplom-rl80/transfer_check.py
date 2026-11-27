"""One predeclared transfer between independently trained early and late action scores."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import json
import sys

import torch
from transfer_adapter import load

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.interface import Move
from maze.environment import Environment, Status

data=import_module("results.E-0016.data")
EXP=PROJECT/"results/E-0016"
OUT=HERE/"transferred-frontier"
digest=lambda p:sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    base_path=EXP/"prev-two/weights/super1500.pt"
    past_path=EXP/"prev-two/weights/super1500.json"
    frontier=HERE/"frontier-search/frozen.json"
    learned=data.read(frontier)["coefficients"]
    assert learned[2:]==[0]*7
    assert digest(base_path)==data.read(past_path)["sha256"]
    bank=EXP/"data/dev.jsonl"
    protocol=dict(mode="unchanged transfer of both nonzero weights selected on whole training episodes",
                  base_sha256=digest(base_path),base_adapter_sha256=digest(EXP/"prev-two/adapter.py"),frontier_sha256=digest(frontier),
                  coefficients=learned[:2],dev_sha256=digest(bank),adapter_sha256=digest(HERE/"transfer_adapter.py"),
                  learn="early base fitted on E-0016 train, late weights fitted on sequence train[1300:1500]",
                  validation="two actual previous readings, same 100 full episodes and a separate ablation; no new dev-selected coefficients")
    assert protocol["base_adapter_sha256"]==data.read(past_path)["adapter_sha256"]
    if (OUT/"protocol.json").exists():assert data.read(OUT/"protocol.json")==protocol
    else:data.save(OUT/"protocol.json",protocol)
    checkpoint=OUT/"model.pt"
    if not checkpoint.exists():
        tmp=OUT/"model.pt.tmp"
        torch.save(dict(format="prev-two-frontier-1",base=torch.load(base_path,weights_only=True,map_location="cpu"),
                        frontier=torch.tensor(learned[:2],dtype=torch.float64)),tmp)
        tmp.replace(checkpoint)
    for ablate in (False,True):
        path=OUT/("ablated.physical.json" if ablate else "dev.physical.json")
        if path.exists():
            r=data.read(path)
            assert r["weights_sha256"]==digest(checkpoint) and r["dev_sha256"]==digest(bank)
            continue
        model=load(checkpoint.read_bytes(),ablate)
        records=[]
        for i,r in enumerate(data.rows(bank)):
            env=Environment(data.level(r))
            cache={}
            actions=bytearray()
            while env.status==Status.RUNNING:
                m=model.choose(env.view,cache)
                actions.append(tuple(Move).index(m))
                env.step(m)
            records.append(dict(map=r["name"],geometry=r["geometry"],tier=r["tier"],outcome=env.status.name,
                                steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions_sha256=sha256(actions).hexdigest()))
            if (i+1)%25==0:print("ablate",ablate,"cases",i+1,sum(e["outcome"]=="SUCCESS" for e in records),flush=True)
        outcomes=Counter(r["outcome"] for r in records)
        result=dict(weights_sha256=digest(checkpoint),dev_sha256=digest(bank),protocol_sha256=digest(OUT/"protocol.json"),
            ablated=ablate,summary=dict(wins=outcomes["SUCCESS"],outcomes=outcomes,
            by_tier={t:sum(e["tier"]==t and e["outcome"]=="SUCCESS" for e in records) for t in ("H1","H2","H3","H4")},
            both=sum(bool(e["refuels"] and e["repairs"]) for e in records)),episodes=records)
        data.save(path,result)
        print(result["summary"],flush=True)


if __name__=="__main__":main()
