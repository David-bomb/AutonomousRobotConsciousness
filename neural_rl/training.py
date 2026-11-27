"""Обновления Double DQN, Rainbow, PPO и управления промежуточными целями."""
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .networks import batch, project
from .rollout import Stage


def tensor(values, dtype=torch.float32):
    return torch.as_tensor(values, dtype=dtype, device="cuda")


class Replay:
    def __init__(self, size):
        self.items = []
        self.priority = np.ones(size, np.float64)
        self.cursor = 0
        self.maximum = 1.

    def add(self, transition):
        if len(self.items) < len(self.priority):
            self.items.append(transition)
        else:
            self.items[self.cursor] = transition
        self.priority[self.cursor] = self.maximum
        self.cursor = (self.cursor+1) % len(self.priority)

    def sample(self, count, rainbow, rng, fraction):
        n = len(self.items)
        if rainbow:
            p = self.priority[:n]**0.6
            p /= p.sum()
            ids = rng.choice(n, size=count, p=p)
            weights = (n*p[ids])**-(0.4+0.6*fraction)
            weights /= weights.max()
        else:
            ids = rng.integers(n, size=count)
            weights = np.ones(count)
        return ids, tensor(weights), zip(*(self.items[i] for i in ids))


def emit(queue, replay):
    state, action = queue[0][:2]
    total, discount = 0., 1.
    for _, _, reward, step_discount, following in queue:
        total += discount*reward
        discount *= step_discount
    replay.add((state, action, total, discount, following))
    queue.popleft()


def optimize(net, optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
    optimizer.step()


def q_update(net, target, optimizer, replay, rng, config, fraction):
    rainbow = net.family == "rainbow"
    ids, weights, samples = replay.sample(config["batch"], rainbow, rng, fraction)
    states, actions, rewards, discounts, following = samples
    data = batch(following)
    index = torch.arange(len(actions), device="cuda")
    moves, rewards, discounts = tensor(actions, torch.long), tensor(rewards), tensor(discounts)
    net.noise()
    target.noise()
    with torch.no_grad():
        chosen = net(data)[0].argmax(1)
        target_q, target_logits, _ = target(data)
        if rainbow:
            distribution = project(target_logits[index, chosen].softmax(-1), rewards, discounts, net.support)
        else:
            expected = rewards + discounts*target_q[index, chosen]
    q, logits, _ = net(batch(states))
    if rainbow:
        logp = logits[index, moves].log_softmax(-1)
        losses = -(distribution*logp).sum(1)
        divergence = (losses.detach() + (distribution*distribution.clamp_min(1e-8).log()).sum(1))
        priority = divergence.clamp_min(0).cpu().numpy()+1e-5
        replay.priority[ids] = priority
        replay.maximum = max(replay.maximum, float(priority.max()))
    else:
        losses = F.smooth_l1_loss(q[index, moves], expected, reduction="none")
    loss = (weights*losses).mean()
    optimize(net, optimizer, loss)
    return loss.detach()


def train_q(net, optimizer, config, pool, rng, np_rng):
    lanes, budget, gamma = config["lanes"], config["ticks"], config["gamma"]
    rainbow = net.family == "rainbow"
    hierarchy = net.family == "hrl_astar"
    updates = steps = 0
    target = deepcopy(net)
    replay = Replay(config["replay"])
    history = []
    for level in (1, 2, 3):
        stage = Stage(pool, level, rng, lanes, hierarchy, config["horizon"])
        queues = [deque() for _ in range(lanes)]
        losses = []
        prior_updates = updates
        while (active := [i for i in range(lanes) if stage.steps[i] < budget]):
            states = [stage.states[i] for i in active]
            with torch.no_grad():
                net.noise()
                actions = net(batch(states))[0].argmax(1).tolist()
            fraction = steps / (3*lanes*budget)
            epsilon = max(0.06, 0.85 - 1.05*fraction)
            for k, (i, state) in enumerate(zip(active, states)):
                action = actions[k]
                if not rainbow and rng.random() < epsilon:
                    action = rng.choice(np.flatnonzero(state.legal))
                reward, discount, following, duration = stage.advance(i, action, budget-stage.steps[i], gamma)
                steps += duration
                queue = queues[i]
                queue.append((state, action, reward, discount, following))
                if len(queue) == (3 if rainbow else 1):
                    emit(queue, replay)
                if discount == 0 or stage.steps[i] == budget:
                    while queue:
                        emit(queue, replay)
            # Равное число обновлений Q на число физических шагов во всех четырёх сериях.
            while len(replay.items) >= config["batch"] and updates < steps//(2*lanes):
                losses.append(q_update(net, target, optimizer, replay, np_rng, config, fraction))
                updates += 1
                if updates % config["sync"] == 0:
                    target.load_state_dict(net.state_dict())
        history.append(dict(stage=level, **stage.report(), updates=updates-prior_updates,
                            mean_loss=torch.stack(losses).mean().item()))
    return history


def gae(rewards, discounts, values, final, lam):
    advantages = torch.empty_like(rewards)
    next_value, running = final, torch.zeros_like(final)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discounts[t]*next_value - values[t]
        running = delta + discounts[t]*lam*running
        advantages[t] = running
        next_value = values[t]
    return advantages


def train_ppo(net, optimizer, config, pool, rng):
    lanes, budget, gamma = config["lanes"], config["ticks"], config["gamma"]
    rollout = config["rollout"]
    history = []
    for level in (1, 2, 3):
        stage = Stage(pool, level, rng, lanes, False, 1)
        losses = []
        for start in range(0, budget, rollout):
            observations, actions, logps, values, rewards, discounts = [], [], [], [], [], []
            with torch.no_grad():
                for tick in range(start, min(budget, start+rollout)):
                    data = batch(stage.states)
                    logits, _, value = net(data)
                    logp = logits.log_softmax(-1)
                    action = torch.multinomial(logp.exp(), 1)
                    rows = [stage.advance(i, a, budget-tick, gamma)
                            for i, a in enumerate(action.flatten().tolist())]
                    observations.append(data)
                    actions.append(action)
                    logps.append(logp.gather(1, action).flatten())
                    values.append(value.flatten())
                    rewards.append([row[0] for row in rows])
                    discounts.append([row[1] for row in rows])
                final = net(batch(stage.states))[2].flatten()
                old_values = torch.stack(values)
                advantages = gae(tensor(rewards), tensor(discounts), old_values, final, config["lambda"])
                returns = (advantages + old_values).flatten()
                advantages = advantages.flatten()
                advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
                old_logp, actions = torch.cat(logps), torch.cat(actions)
                data = tuple(torch.cat(column) for column in zip(*observations))
            for _ in range(config["epochs"]):
                permutation = torch.randperm(len(actions), device="cuda")
                for ids in permutation.split(config["batch"]):
                    logits, _, value = net(tuple(item[ids] for item in data))
                    logp = logits.log_softmax(-1)
                    ratio = (logp.gather(1, actions[ids]).flatten() - old_logp[ids]).exp()
                    clipped = ratio.clamp(1-config["clip"], 1+config["clip"])
                    policy = -torch.minimum(ratio*advantages[ids], clipped*advantages[ids]).mean()
                    entropy = -(logp.exp()*logp).sum(1).mean()
                    loss = policy + 0.5*F.mse_loss(value.flatten(), returns[ids]) - config["entropy"]*entropy
                    optimize(net, optimizer, loss)
                    losses.append(loss.detach())
        history.append(dict(stage=level, **stage.report(), updates=len(losses),
                            mean_loss=torch.stack(losses).mean().item()))
    return history
