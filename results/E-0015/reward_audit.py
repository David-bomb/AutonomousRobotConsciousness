"""Послепроверочный разбор штатной награды. Запуск из корня через -m results.E-0015.reward_audit."""
from collections import Counter
from copy import copy
import gzip
from importlib import import_module
import json
from math import isclose

import torch

from maze.environment import Environment, Status
from maze.grid import destination
from maze.interface import Move
from neural_rl.corpus_study.diagnostic_eval import expected_moves
from neural_rl.corpus_study.learn import setup
from neural_rl.experiment import digest, read, save
from neural_rl.perception import MOVES
from neural_rl.rollout import reward, unknown
from neural_rl.stress_hundreds import ROOT, SOURCE, maps
from scripts.audit_automaton import Reachability
from .evaluate_models import FromFork
from .train import OUT, REG, network

previous = import_module("results.E-0014.investigate")


def branch(env):
    other = copy(env)
    other._memory = copy(env._memory)
    return other


def advance(env, move):
    before = env.view
    view = env.step(move)
    assert env.moved
    return reward(before, view, env.status)


def terms(env, move):
    before = env.view
    alt = branch(env)
    value = advance(alt, move)
    after = alt.view
    d = max(abs(before.dx), abs(before.dy))
    parts = dict(step=-12, goal=35*(d-max(abs(after.dx),abs(after.dy))),
                 memory=8*(unknown(before.memory)-unknown(after.memory)))
    for key, stock, sensor in (("F", "fuel", "fuel_distance"), ("R", "health", "repair_distance")):
        length = getattr(before, sensor)
        active = length is not None and getattr(before, stock) <= max(2*d+7, length+12)
        parts[key] = 55*(length-getattr(after, sensor)) if active else 0
    parts["end"] = 0 if alt.status == Status.RUNNING else 800 if alt.status == Status.SUCCESS else -800
    assert isclose(sum(parts.values())/100, value, abs_tol=1e-9)
    return parts


def fork_rewards():
    with gzip.open(SOURCE/"diagnostics/control.jsonl.gz", "rt", encoding="utf8") as stream:
        records = [r for r in map(json.loads, stream) if r["group"] == 4]
    pool = FromFork(records)
    rows = []
    for case, record in zip(pool.cases, records, strict=True):
        env = pool.start(case)
        view = env.view
        correct, = expected_moves(record, view)
        scores = {m.name:sum(terms(env,m).values()) for m in MOVES
                  if destination(view.window,(1,1),m) is not None}
        name = MOVES[correct].name
        other = max(v for k,v in scores.items() if k != name)
        rows.append(dict(map=case.name,pair=record["pair"],correct=name,reward_cents=scores,
                         rank="higher" if scores[name]>other else "lower" if scores[name]<other else "tie"))
    assert len(rows) == 50
    return rows


def service_delays(cases, witnesses, gamma):
    """Вставить два хода по уже пройденному ребру и доиграть исходный успешный маршрут."""
    def discounted(values):
        return sum(gamma**i * r for i,r in enumerate(values))

    records = []
    for case in cases:
        actions = [MOVES[int(code)] for code in witnesses[case.name]["actions"]]
        env = Environment(case.level)
        original = [advance(env,m) for m in actions]
        assert env.status == Status.SUCCESS
        env.reset()
        for t, move in enumerate(actions, 1):
            advance(env,move)
            x,y = env.position
            tile = env.level.grid[y][x]
            if tile not in "FR":
                continue
            alt = branch(env)
            reverse = Move(tuple(-k for k in move.value))
            suffix = [advance(alt,reverse)]
            if alt.status == Status.RUNNING:
                suffix.append(advance(alt,move))
                assert alt.position == env.position and alt.view.memory == env.view.memory
                for m in actions[t:]:
                    if alt.status != Status.RUNNING:
                        break
                    suffix.append(advance(alt,m))
            # Обе суммы начинаются на сервисе после одного и того же префикса.
            records.append(dict(map=case.name,step=t,tile=tile,outcome=alt.status.name,
                                loop_reward=round(sum(suffix[:2]),6) if alt.steps >= t+2 else None,
                                delta_from_service=round(discounted(suffix)-discounted(original[t:]),6)))
    return records


@torch.inference_mode()
def main():
    reg = read(REG)
    cases = [c for _,c in maps()]
    witnesses = read(ROOT/"results/E-0013/witnesses.json")
    gamma, = {c["gamma"] for c in reg["configs"]}
    oracles = {c.name:Reachability(c.level) for c in cases}
    winning = {c.name:previous.drive(Environment(c.level),map(int,witnesses[c.name]["actions"]),gamma)[1]
               for c in cases}
    report = dict(gamma=gamma, forks=fork_rewards(), service_delays=service_delays(cases,witnesses,gamma), runs={})
    for config in reg["configs"]:
        for arm in reg["arms"]:
            key = config["id"]+"/"+arm
            run = OUT/"runs"/key
            evaluation = read(run/"evaluation.json")
            assert evaluation["sha256"] == digest(run/"weights.pt")
            reference = {r["map"]:r for r in evaluation["stress"]}
            assert len(reference) == len(cases) == 300
            setup(config["seed"])
            net = network(arm,config,reg["input_canvas"]).eval()
            net.load_state_dict(torch.load(run/"weights.pt",weights_only=True,map_location="cuda"))
            result = []
            for start in range(0,len(cases),50):
                chunk = cases[start:start+50]
                traces = previous.action_traces(net,chunk)
                for c in chunk:
                    r = previous.diagnose(c,traces[c.name],oracles[c.name],reference[c.name],winning[c.name],gamma)
                    loss = r["loss"]
                    result.append(dict(map=c.name,step=loss["step"],move=loss["selected"],reward=loss["reward"],
                                       best_viable=loss["best_viable_reward"],policy_return=r["discounted_return"],
                                       witness_return=r["witness_return"]))
                    if key == "double-v03-s2/cnn" and c.name == "H4-04":
                        env = Environment(c.level)
                        for code in traces[c.name][:loss["step"]-1]:
                            advance(env,MOVES[code])
                        details = {a["move"]:terms(env,Move[a["move"]]) for a in loss["choices"]}
                        report["example"] = dict(map=c.name,run=key,choices=loss,terms_cents=details)
            report["runs"][key] = result
            print(key, Counter("unsafe" if r["reward"]>r["best_viable"] else "tie"
                               if r["reward"]==r["best_viable"] else "viable" for r in result),flush=True)
    save(OUT/"reward-audit.json",report)


if __name__ == "__main__":
    main()
