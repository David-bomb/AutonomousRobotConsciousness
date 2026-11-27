"""Обучающие поездки и полная оценка замороженных весов по исходной физике."""
from collections import Counter
from copy import copy
from functools import lru_cache

import torch

from maze.environment import Environment, Status
from .networks import batch
from .perception import command, encode


class MapPool:
    def __init__(self, cases):
        self.cases = cases
        self.templates = {c.name: Environment(c.level) for c in cases}

    def start(self, case):
        env = copy(self.templates[case.name])
        env.reset()
        return env


def curriculum(pool, stage, rng):
    current = [c for c in pool.cases if c.stage == stage]
    previous = [c for c in pool.cases if c.stage < stage]
    while True:
        deck = current + (rng.sample(previous, 10) if previous else [])
        rng.shuffle(deck)
        yield from deck


@lru_cache(maxsize=8192)
def unknown(memory):
    return sum(row.count("?") for row in memory)


def reward(before, after, status):
    offset = max(abs(before.dx), abs(before.dy))
    progress = offset - max(abs(after.dx), abs(after.dy))
    result = -0.12 + 0.35*progress + 0.08*(unknown(before.memory)-unknown(after.memory))
    for stock, sensor in (("fuel", "fuel_distance"), ("health", "repair_distance")):
        available, length = getattr(before, stock), getattr(before, sensor)
        if length is not None and available <= max(2*offset+7, length+12):
            result += 0.55*(length-getattr(after, sensor))
    if status != Status.RUNNING:
        result += 8 if status == Status.SUCCESS else -8
    return result


def execute(env, action, hierarchical, horizon, gamma):
    total, steps = 0., 0
    for move in command(env.view, action, hierarchical, horizon):
        before = env.view
        after = env.step(move)
        total += gamma**steps * reward(before, after, env.status)
        steps += 1
        if env.status != Status.RUNNING:
            break
    if not steps:
        raise RuntimeError("Пустая команда перемещения")
    return total, steps


class Stage:
    def __init__(self, pool, stage, rng, lanes, hierarchical, horizon):
        self.pool = pool
        self.deck = curriculum(pool, stage, rng)
        self.hierarchical, self.horizon = hierarchical, horizon
        self.environments = [self.next_map() for _ in range(lanes)]
        self.states = [encode(env.view, hierarchical) for env in self.environments]
        self.steps = [0]*lanes
        self.outcomes = Counter()
        self.maps = Counter(env.level.title for env in self.environments)
        self.decisions = 0

    def next_map(self):
        return self.pool.start(next(self.deck))

    def advance(self, lane, action, remaining, gamma):
        env = self.environments[lane]
        value, duration = execute(env, int(action), self.hierarchical, min(self.horizon, remaining), gamma)
        done = env.status != Status.RUNNING
        following = encode(env.view, self.hierarchical)
        self.states[lane] = following
        self.steps[lane] += duration
        self.decisions += 1
        if done:
            self.outcomes[env.status.name] += 1
            if duration < remaining:
                env = self.next_map()
                self.maps[env.level.title] += 1
                self.environments[lane] = env
                self.states[lane] = encode(env.view, self.hierarchical)
        return value, 0. if done else gamma**duration, following, duration

    def report(self):
        return dict(steps=sum(self.steps), decisions=self.decisions, outcomes=dict(self.outcomes),
                    maps=dict(self.maps))


@torch.no_grad()
def evaluate(network, pool, horizon):
    network.eval()
    hierarchical = network.family == "hrl_astar"
    active = [(case, pool.start(case), {}) for case in pool.cases]
    rows = []
    while active:
        states = [encode(env.view, hierarchical) for _, env, _ in active]
        actions = network(batch(states))[0].argmax(1).tolist()
        remaining = []
        for (case, env, seen), action in zip(active, actions):
            # При точном повторе снимка и решения воспроизводится тот же цикл.
            # Пропускаем целые периоды до последнего обычного шага симулятора.
            key = env.view
            if key in seen:
                old_steps, old_f, old_r, old_action = seen[key]
                if old_action == action:
                    periods = (env.limit-env.steps-1) // (env.steps-old_steps)
                    env.refuels += periods*(env.refuels-old_f)
                    env.repairs += periods*(env.repairs-old_r)
                    env.steps += periods*(env.steps-old_steps)
            seen[key] = (env.steps, env.refuels, env.repairs, action)
            for move in command(env.view, action, hierarchical, horizon):
                env.step(move)
                if env.status != Status.RUNNING:
                    break
            if env.status == Status.RUNNING:
                remaining.append((case, env, seen))
            else:
                rows.append(dict(map=case.name, stage=case.stage, outcome=env.status.name,
                                 steps=env.steps, refuels=env.refuels, repairs=env.repairs))
        active = remaining
    return sorted(rows, key=lambda r: r["map"])
