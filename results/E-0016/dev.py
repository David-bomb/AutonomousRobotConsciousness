"""Официальная валидация. Каждый ход исполняется в обычной среде."""
import argparse
from collections import Counter
import time
from pathlib import Path

from maze.environment import Environment,Status
from .adapter import Controller
from .data import OUT, digest, level, rows, save


def evaluate(controller,bank):
    episodes=[]
    for row in bank:
        env=Environment(level(row))
        choices={}
        while env.status==Status.RUNNING:
            view=env.view
            if view not in choices:
                choices[view]=controller.choose(view)
            env.step(choices[view])
        episodes.append(dict(map=row["name"],tier=row["tier"],outcome=env.status.name,steps=env.steps,
                             refuels=env.refuels,repairs=env.repairs))
    by_tier=Counter(r["tier"] for r in episodes if r["outcome"]=="SUCCESS")
    summary=dict(wins=sum(by_tier.values()),by_tier={t:by_tier[t] for t in ("H1","H2","H3","H4")},
                 outcomes=dict(Counter(r["outcome"] for r in episodes)),
                 fuel=sum(bool(r["refuels"]) for r in episodes),repair=sum(bool(r["repairs"]) for r in episodes),
                 either=sum(bool(r["refuels"] or r["repairs"]) for r in episodes),
                 both=sum(bool(r["refuels"] and r["repairs"]) for r in episodes))
    return summary,episodes


def verify(path):
    start=time.monotonic()
    bank=OUT/"data/dev.jsonl"
    weight_hash=digest(path)
    summary,episodes=evaluate(Controller(path,weight_hash),rows(bank))
    output=path.with_suffix(".physical.json")
    data=dict(weights_sha256=weight_hash,dev_sha256=digest(bank),summary=summary,episodes=episodes,
              sources={n:digest(OUT/n) for n in ("adapter.py","policy.py","critical_policy.py","rich_policy.py",
                                                  "geometry.cpp","geometry.dll","dev.py")})
    if output.exists():
        import json
        old=json.loads(output.read_bytes())
        assert all(old[k]==v for k,v in data.items() if k!="sources")
    else:
        save(output,dict(**data,seconds=time.monotonic()-start))
    print(path,summary,flush=True)
    return summary,episodes


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("weights",type=Path,nargs="+")
    for checkpoint in parser.parse_args().weights:
        verify(checkpoint)
