"""Classify the first irreversible decisions of pilot200 on its unseen training slice."""
from collections import Counter
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3] / "University/Diplom"
sys.path.insert(0, str(PROJECT))

import torch
from maze.environment import Environment
from maze.grid import destination
from maze.resource_automaton import confirmed_route, known_paths

data = import_module("results.E-0016.data")
policy = import_module("results.E-0016.policy")
teacher = import_module("results.E-0016.teacher")


def main():
    torch.set_num_threads(1)
    exp = PROJECT / "results/E-0016"
    weights = exp / "runs/pilot200/rl.pt"
    old = json.loads((HERE / "pilot-failures.json").read_bytes())
    assert old["checkpoint"] == sha256(weights.read_bytes()).hexdigest()
    by_map = {r["geometry"]: r for r in old["rows"] if "loss" in r}
    net = policy.Policy().eval()
    net.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    predicted = []
    with torch.inference_mode():
        for item in data.rows(exp / "data/train.jsonl")[200:300]:
            prior = by_map.get(item["geometry"])
            if prior is None:
                continue
            env = Environment(data.level(item))
            oracle = teacher.Teacher(env.level)
            known_goal = False
            for step in range(1, prior["loss"] + 1):
                v = env.view
                goal = (v.position[0] + v.dx, v.position[1] + v.dy)
                if v.memory[goal[1]][goal[0]] != "?":
                    known_goal |= confirmed_route(v, goal, known_paths(v.memory, v.position)) is not None
                x, valid = policy.features(v)
                scores = net(torch.from_numpy(x[None]), torch.from_numpy(valid[None]))[0]
                chosen = int(scores.argmax().item())
                if step == prior["loss"]:
                    break
                env.step(policy.MOVES[chosen])
            _, viable = oracle.labels(env)
            assert viable.any() and not viable[chosen], (item["name"], step)
            targets = ("F", "R", "B")
            measurement = [v.fuel_distance, v.repair_distance, oracle.dist[4][v.position]]
            reducing = {target: [] for target in targets}
            for idx, move in enumerate(policy.MOVES):
                q = destination(v.window, (1, 1), move)
                if q is None:
                    continue
                pt = (v.position[0] + move.value[0], v.position[1] + move.value[1])
                after = [min(d[pt] for d in oracle.dist[:2]), min(d[pt] for d in oracle.dist[2:4]), oracle.dist[4][pt]]
                for target, d0, d1 in zip(targets, measurement, after):
                    if d1 < d0:
                        reducing[target].append(idx)
            # "Urgent" is fixed by ordinary current readings. Ties are explicit.
            slacks = [v.fuel - v.fuel_distance, v.health - v.repair_distance]
            urgent = [target for target, slack in zip(targets[:2], slacks) if slack == min(slacks)]
            urgent_moves = {i for target in urgent for i in reducing[target]}
            good = set(viable.nonzero()[0].tolist())
            predicted.append(dict(geometry=item["geometry"], tier=item["tier"], first_loss=prior["loss"],
                                  phase="both" if prior["after_both"] else "neither" if prior["before_first"] else "one",
                                  urgent=urgent, slacks=slacks, bad_move=chosen, viable=sorted(good),
                                  reducing=reducing, urgent_gradient_available=bool(urgent_moves & good),
                                  urgent_gradient_missed=bool(urgent_moves & good) and chosen not in urgent_moves,
                                  urgent_gradient_sufficient=bool(urgent_moves) and urgent_moves <= good,
                                  ever_confirmed_goal=known_goal))
    assert len(predicted) == len(by_map)
    groups = {}
    for phase in ["all", "neither", "one", "both"]:
        group = [r for r in predicted if phase == "all" or r["phase"] == phase]
        counts = Counter({k: sum(r[k] for r in group) for k in (
            "urgent_gradient_available", "urgent_gradient_missed", "urgent_gradient_sufficient", "ever_confirmed_goal")})
        groups[phase] = dict(n=len(group), **counts)
    result = dict(split="E-0016 train[200:300] outside pilot200 training, no dev or controls",
                  definition="at first irreversible transition of resource-failed episodes only; actual neighbor distances used offline",
                  dirs=[m.name for m in policy.MOVES], checkpoint=old["checkpoint"], groups=groups, records=predicted)
    out = HERE / "irreversible-choices.json"
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if out.exists() and out.read_bytes() != encoded:
        raise ValueError("Archived result differs")
    out.write_bytes(encoded)
    print(json.dumps(groups))


if __name__ == "__main__":
    main()
