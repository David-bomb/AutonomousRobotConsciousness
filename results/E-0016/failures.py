"""Разбор первого потерянного финиша только на train."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from maze.environment import Environment,Status
from scripts.audit_automaton import Reachability
from .adapter import Controller
from .data import OUT, digest, level, rows, save
from .policy import MOVES
from .teacher import Teacher


def main(path):
    net=Controller(path,digest(path))
    report=[]
    for r in rows(OUT/"data/train.jsonl")[:100]:
        env=Environment(level(r))
        teacher=Teacher(env.level)
        oracle=Reachability(env.level)
        first=None
        choices={}
        deviations=0
        reversals=0
        positions=[env.position]
        for _ in range(env.limit):
            view=env.view
            if view not in choices:
                choices[view]=net.choose(view)
            move=choices[view]
            if first is None:
                shortest,viable=teacher.labels(env)
                deviations+=int(not shortest[MOVES.index(move)])
                if not viable[MOVES.index(move)]:
                    alternatives=[m.name for m,q in oracle.edges[env.position]
                                  if oracle.possible(q,*oracle.after(q,env.fuel,env.health),env.limit-env.steps-1)]
                    if alternatives and move.name not in alternatives:
                        first=dict(step=env.steps+1,refuels=env.refuels,repairs=env.repairs,viable=alternatives,
                                   selected=move.name,fuel=env.fuel,health=env.health,
                                   slack_f=env.fuel-view.fuel_distance,slack_r=env.health-view.repair_distance)
            env.step(move)
            if len(positions)>1 and env.position==positions[-2]:
                reversals+=1
            positions.append(env.position)
            if env.status!=Status.RUNNING:
                break
        report.append(dict(map=r["name"],outcome=env.status.name,first=first,steps=env.steps,
                           deviations_before_loss=deviations,reversals=reversals,unique_positions=len(set(positions))))
        teacher.remaining.cache_clear()
    losses=[r["first"] for r in report if r["first"]]
    summary=dict(outcomes=dict(Counter(r["outcome"] for r in report)),localized=len(losses),
                 before_first=sum(not (l["refuels"] or l["repairs"]) for l in losses),
                 after_both=sum(bool(l["refuels"] and l["repairs"]) for l in losses),
                 median_loss_step=median(l["step"] for l in losses) if losses else None,
                 repeated_edge_fraction_median=median(r["reversals"]/r["steps"] for r in report))
    save(path.with_suffix(".train-diagnosis.json"),dict(weights_sha256=digest(path),summary=summary,episodes=report))
    print(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("weights",type=Path)
    main(p.parse_args().weights)
