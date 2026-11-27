"""Парный повтор с исключением всех 200 внутренних геометрий и прежним RL-выходом."""
import time
import torch
from ...data import OUT, digest, read, rows, save
from ...dev import evaluate
from .model import HERE
from .transfer_model import load


def main():
    original=HERE.parent/"runs/field3200/rl.pt"
    folder=HERE/"field-3000-independent"
    step=read(folder/"results.json")["selected"]
    corrected=folder/f"s{step}.pt"
    out=HERE.parent/"runs/field3000-clean"
    out.mkdir(exist_ok=False)
    pack=torch.load(original,weights_only=True,map_location="cpu")
    pack["field"]=torch.load(corrected,weights_only=True,map_location="cpu")
    p=out/"rl.pt"
    torch.save(pack,p)
    d=OUT/"data/dev.jsonl"
    save(out/"locked-before-dev.json",dict(weights_sha256=digest(p),base=digest(original),
              field_sha256=digest(corrected),selection="original terminal weights fixed",dev_sha256=digest(d)))
    start=time.monotonic()
    summary,episodes=evaluate(load(p.read_bytes()),rows(d))
    save(p.with_suffix(".physical.json"),dict(weights_sha256=digest(p),dev_sha256=digest(d),
             summary=summary,episodes=episodes,seconds=time.monotonic()-start))
    print(summary,flush=True)


if __name__=="__main__":
    main()
