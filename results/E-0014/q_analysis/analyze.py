"""Аудит архивных журналов Q и решений на развилках физически успешных маршрутов."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from maze.environment import Environment, Status
from maze.grid import destination, wave
from neural_rl import perception
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read
from neural_rl.networks import Network, batch
from neural_rl.rollout import evaluate
from neural_rl.stress_hundreds import maps, MOVES

SRC = ROOT / "results/E-0012"
HUNDREDS = ROOT / "results/E-0013"


def journals(family):
    stages = defaultdict(Counter)
    variants = defaultdict(list)
    seen = defaultdict(list)
    screens = defaultdict(Counter)
    hard_wins = []
    for run_file in sorted((SRC / "runs").glob(f"{family}-*/run.json")):
        run = read(run_file)
        conf = run["config"]
        assert run["train_steps"] == 73728 and run["updates"] == 1536 and run["parameter_delta"] > 0
        variants[conf["variant"]].append(run["screen"]["wins"])
        seen[conf["geometry_count"]].append(run["train_geometries"])
        validation = read(run_file.with_name("validation.json"))
        for when, rows in validation.items():
            for r in rows:
                screens[(when, r["stage"])].update(episodes=1, wins=int(r["outcome"] == "SUCCESS"))
        for stage in run["training"]:
            n = stage["stage"]
            counters = stages[n]
            finished = sum(stage["outcomes"].values())
            success = stage["outcomes"].get("SUCCESS", 0)
            counters.update(steps=stage["steps"], updates=stage["updates"], completed=finished,
                            wins=success, runs_with_no_success=int(success == 0),
                            target_syncs=stage["updates"] // conf["sync"])
            if n == 3:
                hard_wins.append(success)
    assert len(hard_wins) == 50
    return dict(training_stages=stages, screen={f"{t}_{n}": d for (t,n), d in screens.items()},
                stage3_wins_range=[min(hard_wins), median(hard_wins), max(hard_wins)],
                variant_wins_on_screen=variants, geometries_seen_range={k:[min(v),max(v)] for k,v in seen.items()})


def networks(family):
    model = read(SRC / "frozen.json")[family]["id"]
    folder = SRC / "runs" / model
    run = read(folder / "run.json")
    assert digest(folder / "weights.pt") == run["weights_sha256"]
    config = run["config"]
    setup(config["seed"])
    initial = Network(config).cuda().eval()
    val = Pool("validation_id",60)
    replay = [r for i in range(0,240,32) for r in evaluate(initial,Chunk(val,val.cases[i:i+32]),1)]
    assert replay == read(folder / "validation.json")["before"]
    trained = Network(config).cuda().eval()
    trained.load_state_dict(torch.load(folder / "weights.pt",weights_only=True,map_location="cuda"))
    return model, dict(before=initial, after=trained)


def junctions(case, proof):
    env = Environment(case.level)
    grid, goal = case.level.grid, case.level.goal
    distance_to_goal = {p:d for p,d,_ in wave(grid,[goal])}
    for a in proof["actions"]:
        pos = env.position
        road = distance_to_goal[pos]
        if env.refuels and env.repairs and min(env.fuel,env.health) >= road:
            s = perception.encode(env.view,False)
            if s.legal.sum() >= 3:
                successors = {i:destination(grid,pos,MOVES[i]) for i in np.flatnonzero(s.legal)}
                optimal = {i for i,p in successors.items() if distance_to_goal[p] < road}
                closer = {i for i,p in successors.items() if perception.distance(p,goal) < perception.distance(pos,goal)}
                group = "agree" if closer & optimal else "opposed" if closer else "no_geometric_progress_available"
                # На копии убираем только ранее раскрытые стены и дороги вне окна.
                img = s.image.copy()
                w,h = len(grid[0]),len(grid)
                known = ((img[:h,:w] & (4|2)) != 0) & ((img[:h,:w] & (8|16)) == 0)
                x,y = pos
                known[max(0,y-1):y+2,max(0,x-1):x+2] = False
                img[:h,:w][known] = (img[:h,:w][known] & 32) | 1
                yield case.name, group, s, s._replace(image=img), optimal, closer, int(known.sum())
        env.step(MOVES[int(a)])
        assert env.moved
    assert env.status == Status.SUCCESS


@torch.no_grad()
def inspect_batch(rows, models, records):
    real = batch([s for _,_,s,_,_,_,_ in rows])
    erased = batch([s for _,_,_,s,_,_,_ in rows])
    for name, nets in models.items():
        a0 = nets["before"](real)[0].argmax(1).tolist()
        a1 = nets["after"](real)[0].argmax(1).tolist()
        no_far = nets["after"](erased)[0].argmax(1).tolist()
        for (case, group, state, _, optimal, closer, removed), first, after, cut in zip(rows,a0,a1,no_far):
            d = records[name][case][group]
            d.update(n=1, best_before=int(first in optimal),best_after=int(after in optimal),
                     chance=len(optimal)/int(state.legal.sum()), geometry_before=int(first in closer),
                     geometry_after=int(after in closer),memory_changed=int(cut!=after),
                     best_after_without_far=int(cut in optimal), cells_removed=removed,
                     different_input=int(removed>0))


def main():
    models = dict(networks(family) for family in ("double","dueling"))
    records = {name:defaultdict(lambda: defaultdict(Counter)) for name in models}
    evidence = read(HUNDREDS / "witnesses.json")
    rows = []
    for _, case in maps():
        for row in junctions(case,evidence[case.name]):
            rows.append(row)
            if len(rows)==128:
                inspect_batch(rows,models,records)
                rows.clear()
    if rows:
        inspect_batch(rows,models,records)
    summaries = {}
    for name, grouped in records.items():
        summary = defaultdict(Counter)
        for groups in grouped.values():
            for group, counts in groups.items():
                summary[group].update(counts)
                summary[group]["maps"] += 1
        summaries[name] = summary
    stats=dict(journals={f:journals(f) for f in ("double","dueling")},branches=summaries,
               reproduced_before_per_model=240, per_map=records)
    (HERE/"metrics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=1),encoding="utf-8")
    print(json.dumps({k:v for k,v in stats.items() if k!="per_map"},ensure_ascii=False,indent=1))


if __name__ == "__main__":
    main()
