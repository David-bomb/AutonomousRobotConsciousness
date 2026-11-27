"""Частота учебных финишей и проверка трёх способов увеличить награду за поездку."""
from collections import Counter
from itertools import cycle, islice
from math import isclose

import torch

from maze.environment import Environment, Status
from maze.grid import cell, destination
from maze.interface import Move
from neural_rl.corpus_study.learn import setup
from neural_rl.experiment import digest, read, save
from neural_rl.perception import MOVES
from neural_rl.rollout import unknown
from neural_rl.stress_hundreds import ROOT, maps
from .reward_audit import advance, branch, previous, terms
from .train import OUT, REG, network


def discounted(rewards, gamma):
    value = 0.
    for r in reversed(rewards):
        value = r + gamma*value
    return value


def opposite(move):
    dx,dy = move.value
    return Move((-dx,-dy))


def drive(env, moves, gamma, *, period=None, observe_cycle=False):
    rewards, discovery, seen = [], [], {}
    entry = length = None
    for i,move in enumerate(moves):
        if env.status != Status.RUNNING:
            break
        if entry is None and (observe_cycle or (period is not None and i%period == 0)):
            if env.view in seen:
                entry = seen[env.view]
                length = i-entry
            else:
                seen[env.view] = i
        before = env.view
        rewards.append(advance(env,move))
        discovery.append(.08*(unknown(before.memory)-unknown(env.view.memory)))
    assert env.status != Status.RUNNING
    row = dict(outcome=env.status.name,steps=len(rewards),total=discounted(rewards,gamma),
               memory=discounted(discovery,gamma))
    if entry is not None:
        period_rewards = rewards[entry:entry+length]
        assert len(period_rewards)==length and sum(discovery[entry:])==0
        # Каждый последующий ход сверяется с периодом, включая штраф за лимит.
        assert env.status == Status.LIMIT
        assert all(isclose(value,period_rewards[(k-entry)%length]-(8 if k==len(rewards)-1 else 0),abs_tol=1e-8)
                   for k,value in enumerate(rewards) if k>=entry+length)
        row.update(entry=entry,period=length,period_reward=round(sum(period_rewards),6))
    return row


def detour(env, path_positions):
    if cell(env.level.grid,*env.position) in "FR":
        return None
    for move in MOVES:
        p = destination(env.level.grid,env.position,move)
        if p is None or p in path_positions or cell(env.level.grid,*p) in "FR":
            continue
        alt = branch(env)
        before = unknown(env.view.memory)
        advance(alt,move)
        if alt.status == Status.RUNNING and unknown(alt.view.memory)<before:
            return move
    return None


def scripted(case, moves, gamma):
    env = Environment(case.level)
    positions = [env.position]
    sites = []
    rewards = []
    for i,move in enumerate(moves,1):
        rewards.append(advance(env,move))
        positions.append(env.position)
        tile = cell(env.level.grid,*env.position)
        if tile in "FR":
            sites.append((i,tile))
    assert env.status == Status.SUCCESS
    first,kind = sites[0]
    second = next(i for i,tile in sites if tile!=kind)
    # Одна и та же дорога между первыми достигнутыми сервисами двух типов.
    route = moves[first:second]
    loop = [opposite(m) for m in reversed(route)]+route
    env.reset()
    for move in moves[:second]:
        advance(env,move)
    shuttle = drive(branch(env),cycle(loop),gamma,period=len(loop))
    shuttle.update(from_step=second,witness=discounted(rewards[second:],gamma))
    if shuttle.get("period_reward",0)>0:
        commands = cycle(loop)
        for move in islice(commands,shuttle["entry"]):
            advance(env,move)
        breakdown = Counter()
        for move in islice(commands,shuttle["period"]):
            breakdown.update(terms(env,move))
            advance(env,move)
        assert isclose(sum(breakdown.values())/100,shuttle["period_reward"],abs_tol=1e-8)
        shuttle["period_terms_cents"] = dict(breakdown)

    env.reset()
    distinct = set(positions)
    for i,move in enumerate(moves):
        side = detour(env,distinct)
        if side is not None:
            original = drive(branch(env),moves[i:],gamma)
            longer = drive(branch(env),[side,opposite(side),*moves[i:]],gamma)
            local = branch(env)
            parts = Counter(terms(local,side))
            advance(local,side)
            parts.update(terms(local,opposite(side)))
            return dict(map=case.name,shuttle=shuttle,detour=dict(step=i,direction=side.name,
                        two_step_cents=dict(parts),original=original,longer=longer))
        advance(env,move)
    return dict(map=case.name,shuttle=shuttle,detour=None)


@torch.inference_mode()
def main():
    reg = read(REG)
    gamma, = {c["gamma"] for c in reg["configs"]}
    cases = {c.name:c for _,c in maps()}
    witnesses = read(ROOT/"results/E-0013/witnesses.json")
    data = dict(gamma=gamma,training={},natural=[],scripted=[])
    for config in reg["configs"]:
        for arm in reg["arms"]:
            key = config["id"]+"/"+arm
            folder = OUT/"runs"/key
            history = read(folder/"training.json")["history"]
            data["training"][key] = [dict(stage=s["stage"],steps=s["steps"],started=sum(s["maps"].values()),
                                          finished=sum(s["outcomes"].values()),wins=s["outcomes"].get("SUCCESS",0))
                                     for s in history]
            evaluation = read(folder/"evaluation.json")
            limited = [r for r in evaluation["stress"] if r["outcome"]=="LIMIT"]
            if not limited:
                continue
            setup(config["seed"])
            assert digest(folder/"weights.pt") == evaluation["sha256"]
            net = network(arm,config,reg["input_canvas"]).eval()
            net.load_state_dict(torch.load(folder/"weights.pt",map_location="cuda",weights_only=True))
            traces = previous.action_traces(net,[cases[r["map"]] for r in limited])
            for r in limited:
                env = Environment(cases[r["map"]].level)
                actual = drive(env,(MOVES[a] for a in traces[r["map"]]),gamma,observe_cycle=True)
                assert (actual["outcome"],actual["steps"],env.refuels,env.repairs)==tuple(
                    r[k] for k in ("outcome","steps","refuels","repairs"))
                data["natural"].append(dict(run=key,map=r["map"],**actual))
    for name,case in cases.items():
        moves = [MOVES[int(code)] for code in witnesses[name]["actions"]]
        data["scripted"].append(scripted(case,moves,gamma))
    save(OUT/"reward-routes.json",data)
    print(len(data["training"]),"обучений,",len(data["natural"]),"поездок до лимита,",
          Counter(r["shuttle"]["outcome"] for r in data["scripted"]),flush=True)


if __name__=="__main__":
    main()
