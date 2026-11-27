"""Generate prospective transfer banks, exposing only geometry fingerprints of protected data."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
import sqlite3
import sys
from renewable_train import HERE,EXP,PROJECT,data,digest,save
from maze.datasets import geometry


def main():
    out=HERE/"combination-search";out.mkdir(exist_ok=True)
    method=PROJECT/"results/verification-rl80/wave3-selection"
    sys.path.insert(0,str(method))
    from banks import keys
    registry=data.read(EXP/"sequence/spatial/bank-audit.json")
    used=import_module("results.E-0016.sequence.spatial.fresh").fingerprints()
    for entry in registry["banks"]:
        path=EXP/entry["path"]
        assert digest(path)==entry["sha256"]
        used.update(keys(path))
    assert len(used)==registry["combined_count"]
    with sqlite3.connect((EXP/"factorial/train-dev.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        used.update(h for h, in db.execute("select hash from geoms"))
    for p in EXP.rglob("*"):
        if p.is_file() and p.suffix in (".json",".jsonl",".gz") and not p.name.startswith("."):
            used.update(keys(p))
    paths=[HERE/x for x in ("handover-search/train.jsonl","return-search/bank.jsonl","panel-search/holdout.jsonl")]
    paths.extend([method/"9167001.jsonl",method/"9167002.jsonl",method.parent/"phase100-train/train.jsonl"])
    for p in paths:
        used.update(keys(p))
    original=len(used);prior=sha256("\n".join(sorted(used)).encode()).hexdigest()
    plan=HERE/"joint_selection_plan.json"
    sets={}
    for seed in data.read(plan)["seeds"]:
        dest=out/(str(seed)+".jsonl")
        entries=data.rows(dest) if dest.exists() else data.generate(("joint-transfer",seed))
        counts=Counter(r["tier"] for r in entries)
        shapes={r["geometry"] for r in entries}
        assert counts==dict.fromkeys(("H1","H2","H3","H4"),25) and len(shapes)==100
        assert all(geometry(r["grid"])==r["geometry"] for r in entries) and not shapes&used
        if not dest.exists():
            temp=dest.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in entries),encoding="utf8")
            temp.replace(dest)
        sets[str(seed)]=dict(sha256=digest(dest),prior_overlap=0,tiers=dict(counts))
        used.update(shapes)
    receipt=out/"banks.json"
    record=dict(seeds=sets,excluded_count=original,excluded_sha256=prior,
                rule_sha256=digest(plan),generator_sha256=digest(data.PACK/"make_hard_maps.py"))
    if receipt.exists():
        assert data.read(receipt)["seeds"]==sets
    else:
        save(receipt,record)
    print(record,flush=True)


if __name__=="__main__":main()
