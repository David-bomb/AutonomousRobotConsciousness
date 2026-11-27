"""Посмертный разбор E-0013. Запуск из корня через python -m results.E-0014.investigate."""
from copy import copy
from dataclasses import replace
from pathlib import Path
from statistics import median

import torch

from maze.environment import Environment, Status
from neural_rl.corpus_study.learn import setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import MOVES, encode
from neural_rl.rollout import MapPool, evaluate, reward
from neural_rl.stress_hundreds import ROOT, SOURCE, maps
from scripts.audit_automaton import Reachability

OUT = Path(__file__).resolve().parent
STRESS = ROOT / "results/E-0013"


@torch.inference_mode()
def action_traces(net, cases):
    active = [(c.name, Environment(c.level), [], {}) for c in cases]
    complete = {}
    while active:
        actions = net(batch([encode(env.view, False) for _, env, _, _ in active]))[0].argmax(1).tolist()
        pending = []
        for (name, env, moves, seen), action in zip(active, actions):
            old = seen.get(env.view)
            if old is not None and moves[old] == action:
                period = moves[old:]
                full, tail = divmod(env.limit-env.steps, len(period))
                complete[name] = moves + period * full + period[:tail]
                continue
            seen[env.view] = env.steps
            moves.append(action)
            env.step(MOVES[action])
            if env.status == Status.RUNNING:
                pending.append((name, env, moves, seen))
            else:
                complete[name] = moves
        active = pending
    return complete


def drive(env, moves, gamma):
    states = [(env.position, env.fuel, env.health)]
    first_service = None
    total = 0.
    for step, action in enumerate(moves, 1):
        before = env.view
        env.step(MOVES[action])
        assert env.moved
        states.append((env.position, env.fuel, env.health))
        total += gamma**(step-1) * reward(before, env.view, env.status)
        if first_service is None and (env.refuels or env.repairs):
            first_service = step
    return states, total, first_service


def diagnose(case, actions, oracle, reference, win_return, gamma):
    env = Environment(case.level)
    states, total, first_service = drive(env, actions, gamma)
    actual = dict(outcome=env.status.name, steps=env.steps, refuels=env.refuels, repairs=env.repairs)
    assert all(reference[k] == v for k,v in actual.items())
    assert env.status != Status.SUCCESS

    def possible(t):
        return oracle.possible(*states[t], env.limit-t)

    lo, hi = 0, len(actions)
    while hi-lo > 1:
        mid = (lo+hi)//2
        if possible(mid):
            lo = mid
        else:
            hi = mid
    assert possible(lo) and not possible(hi)
    env.reset()
    for action in actions[:lo]:
        env.step(MOVES[action])
    before = env.view
    choices = []
    for move, _ in oracle.edges[before.position]:
        alt = copy(env)
        alt._memory = copy(env._memory)  # Снимок с независимым обновлением открытых клеток.
        alt.step(move)
        choices.append(dict(move=move.name, possible=oracle.possible(
            alt.position, alt.fuel, alt.health, alt.limit-alt.steps),
            reward=round(reward(before,alt.view,alt.status),6)))
    bad = next(x for x in choices if x["move"] == MOVES[actions[lo]].name)
    viable = [x["reward"] for x in choices if x["possible"]]
    assert not bad["possible"] and viable
    positions = [s[0] for s in states]
    reversals = sum(p == r for p,r in zip(positions,positions[2:]))
    return dict(map=case.name, **actual, actions="".join(map(str,actions)),
                discounted_return=total, witness_return=win_return, first_service=first_service,
                immediate_returns=reversals, unique_positions=len(set(positions)),
                loss=dict(step=hi, position=before.position, fuel=before.fuel, health=before.health,
                          dF=before.fuel_distance, dR=before.repair_distance, selected=bad["move"],
                          reward=bad["reward"], best_viable_reward=max(viable), choices=choices,
                          after_service=bool(env.refuels or env.repairs)))


class PrefixPool(MapPool):
    """Движение штатного робота по сохранённому маршруту, затем сеть получает управление."""
    def __init__(self, cases, witnesses):
        super().__init__(cases)
        self.witnesses = witnesses

    def start(self, case):
        env = super().start(case)
        actions = self.witnesses[case.name]["actions"]
        for code in actions[:-5]:
            env.step(MOVES[int(code)])
        assert env.status == Status.RUNNING and len(actions)>=5
        return env


def run():
    manifest = read(STRESS/"manifest.json")
    witnesses = read(STRESS/"witnesses.json")
    cases = [case for _,case in maps()]
    chunks = [cases[i:i+50] for i in range(0,len(cases),50)]
    assert len(witnesses)==len(cases)==300
    frozen = read(SOURCE/"frozen.json")
    oracles = {case.name: Reachability(case.level) for case in cases}
    winning = {}
    for family, record in frozen.items():
        name = record["id"]
        model = manifest["models"][name]
        assert digest(ROOT/model["weights"]) == record["sha256"] == model["sha256"]
        setup(model["config"]["seed"])
        net = Network(model["config"]).cuda().eval()
        net.load_state_dict(torch.load(ROOT/model["weights"],map_location="cuda",weights_only=True))
        gamma = model["config"]["gamma"]
        if gamma not in winning:
            winning[gamma] = {}
            for case in cases:
                env = Environment(case.level)
                _, value, _ = drive(env, map(int,witnesses[case.name]["actions"]),gamma)
                assert env.status == Status.SUCCESS
                winning[gamma][case.name] = value
        reference = {r["map"]:r for r in read(STRESS/"models"/(name+".json"))["rows"]}
        rows = []
        for chunk in chunks:
            actions = action_traces(net,chunk)
            rows.extend(diagnose(c,actions[c.name],oracles[c.name],reference[c.name],
                                 winning[gamma][c.name],gamma) for c in chunk)
        save(OUT/(family+"-traces.json"),rows)
        print(name, "verified",len(rows),"median loss",median(r["loss"]["step"] for r in rows),flush=True)
        controls = {}
        for condition, stocks in (("fuel100",{"fuel":100}),("health100",{"health":100}),
                                  ("both100",{"fuel":100,"health":100}),("last5",{})):
            pool_rows = []
            for chunk in chunks:
                pool = (MapPool([replace(c,level=replace(c.level,**stocks)) for c in chunk])
                        if stocks else PrefixPool(chunk,witnesses))
                pool_rows.extend(evaluate(net,pool,1))
            assert len({r["map"] for r in pool_rows})==300
            controls[condition]=pool_rows
            print(name,condition,sum(r["outcome"]=="SUCCESS" for r in pool_rows),flush=True)
        save(OUT/(family+"-controls.json"),controls)


if __name__ == "__main__":
    run()
