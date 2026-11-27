"""Audit training labels at the real initial state of each new training maze. No policy is evaluated."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def audit(project):
    sys.path.insert(0, str(project))
    from maze.levels import Level
    from scripts.audit_automaton import Reachability

    data = project / "results/E-0016/data"
    manifest = json.loads((data/"manifest.json").read_bytes())
    raw = (data/"train.jsonl").read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if fingerprint != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training split hash mismatch")

    records = []
    counts = {tier: Counter() for tier in ("H1", "H2", "H3", "H4")}
    for line in raw.splitlines():
        row = json.loads(line)
        level = Level.parse("\n".join([row["name"], f'{row["fuel"]} {row["health"]}', *row["grid"]]))
        oracle = Reachability(level)
        remaining = 8 * len(level.grid) * len(level.grid[0]) - 1
        viable, shortest, legal = [], [], []
        for move, q in oracle.edges[level.start]:
            f, h = oracle.after(q, level.fuel, level.health)
            legal.append(move.name)
            if oracle.possible(q, f, h, remaining):
                viable.append(move.name)
                if oracle.possible(q, f, h, row["optimum"] - 1):
                    shortest.append(move.name)
        if not shortest:
            raise ValueError(f'No start action meets the known route length on {row["name"]}')
        records.append(dict(geometry=row["geometry"], tier=row["tier"], legal=legal,
                            can_still_finish=viable, can_match_optimum=shortest))
        c = counts[row["tier"]]
        c["maps"] += 1
        c["multiple_viable"] += len(viable) > 1
        c["any_lethal"] += len(viable) < len(legal)
        c["viable_nonoptimal"] += len(viable) > len(shortest)
        c["any_optimal_tie"] += len(shortest) > 1
        c["legal_moves"] += len(legal)
        c["viable_moves"] += len(viable)
        c["shortest_moves"] += len(shortest)
    return dict(split="train", sha256=fingerprint, unit="first move of each full episode",
                tiers=counts, records=records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    result = audit(a.project.resolve())
    body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
    if a.output.exists():
        if a.output.read_bytes() != body:
            raise ValueError("Existing output differs")
    else:
        a.output.write_bytes(body)
    print(json.dumps(result["tiers"], indent=2))
