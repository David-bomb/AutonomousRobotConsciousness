import json
from hashlib import sha256
from maze.datasets import geometry
from ...data import OUT, digest, generate, rows, save
from .fresh import HERE, fingerprints


def main():
    used=fingerprints()
    for file in (OUT/"data/train.jsonl",OUT/"data/dev.jsonl",OUT/"data/confirmation.jsonl",
                 HERE.parent/"train.jsonl",HERE.parent/"probe.jsonl",HERE/"extra.jsonl",HERE/"selection.jsonl",
                 HERE/"large.jsonl",HERE/"reward-screen.jsonl",HERE/"reward-selection.jsonl",HERE/"reward-probe.jsonl"):
        used.update(row["geometry"] for row in rows(file))
    exclusions=sha256("\n".join(sorted(used)).encode()).hexdigest()
    dest=HERE/"gate-selection.jsonl"
    seeds=[9164200,9164201]
    with dest.open("x",encoding="utf8",newline="\n") as f:
        for seed in seeds:
            for r in generate(("gate-select",seed)):
                key=geometry(r["grid"])
                assert key not in used and key==r["geometry"]
                used.add(key)
                f.write(json.dumps(r,separators=(",",":"))+"\n")
    save(HERE/"gate-selection.json",dict(rows=200,seeds=seeds,excluded_sha256=exclusions,sha256=digest(dest)))


if __name__=="__main__":main()
