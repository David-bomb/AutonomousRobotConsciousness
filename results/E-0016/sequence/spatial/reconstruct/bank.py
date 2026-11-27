from collections import Counter
from hashlib import sha256
from pathlib import Path

from maze.datasets import geometry
from ....data import OUT, digest, read, rows, save

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / "runs/reconstruct400"
CACHE = HERE.parent.parent / "cache/reconstruct"
PACK = OUT / "prev-two/early/wave1-joint400.pt"
DEV = OUT / "data/dev.jsonl"
RANGES = {"fit": (7000, 7400), "held": (7400, 7500), "selection": (7500, 7600)}


def maps(split):
    start, end = RANGES[split]
    return rows(HERE.parent / "large.jsonl")[start:end]


def main():
    bank = HERE.parent / "large.jsonl"
    assert digest(bank) == read(HERE.parent / "large.json")["sha256"]
    assert digest(PACK) == read(PACK.with_suffix(".json"))["sha256"]
    archive = read(HERE.parent / "early.json")
    assert archive["sets"][0]["maps"] == 3000
    used = {r["geometry"] for r in rows(HERE.parent / "extra.jsonl")[:3000]}
    used.update(r["geometry"] for r in rows(DEV))
    groups = {}
    for split in RANGES:
        data = maps(split)
        hashes = [geometry(r["grid"]) for r in data]
        assert hashes == [r["geometry"] for r in data]
        assert len(set(hashes)) == len(data) and not set(hashes) & used
        used.update(hashes)
        tiers = Counter(r["tier"] for r in data)
        assert set(tiers.values()) == {len(data) // 4}
        groups[split] = dict(span=RANGES[split], tiers=tiers,
                             geometry=sha256("\n".join(hashes).encode()).hexdigest())
    target = HERE / "registration.json"
    spec = dict(bank=digest(bank), groups=groups, pair_archive=digest(HERE.parent / "early.json"),
                frozen=digest(PACK), dev=digest(DEV), protocol=digest(HERE / "protocol.md"),
                selection_order=["marginal1", "direct1", "missing", "marginal0.5", "direct0.5"],
                collection=dict(max_early_steps=80, max_samples=36, seed=20377))
    if target.exists():
        import json
        assert read(target) == json.loads(json.dumps(spec))
    else:
        save(target, spec)
    RUN.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    print(groups, flush=True)


if __name__ == "__main__":
    main()
