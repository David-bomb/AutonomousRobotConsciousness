"""Зависимость восьми выходов замороженной wave1 от предполагаемой пары."""
from importlib import import_module
from itertools import product

import numpy as np
import torch

from maze.interface import Move
from ..local_model import local_image

legacy = import_module("results.E-0016.prev-two.adapter")
phase = import_module("results.E-0016.prev-two.early_policy").complete
MOVES = tuple(Move)
REVERSE = (2, 3, 0, 1, 6, 7, 4, 5, 8)
DELTAS = np.array(list(product((-1, 0, 1), repeat=2)), dtype=np.int8)


def early_model(state):
    kind = "rich" if "base.base.wall_prior" in state else "critical" if "base.base.risk.0.weight" in state else "plain"
    net = legacy.PrevNet(kind).eval().requires_grad_(False)
    net.load_state_dict(state)
    return net


def sign_index(previous, current):
    if previous is None:
        return 9
    f, r = np.asarray(current) - previous
    assert max(abs(f), abs(r)) <= 1
    return int(3 * (f + 1) + r + 1)


@torch.inference_mode()
def frame(net, view):
    core, gradient, legal = net.compressed(view)
    unknown, bounds = legacy.memory_fields(view, (None, None), legal)
    values = [(core, gradient, unknown, bounds, legal)]
    allowed = np.zeros(9, dtype=bool)
    last = MOVES.index(view.last_move) if view.last_move is not None else 8
    current = np.array([view.fuel_distance, view.repair_distance])
    if last != 8:
        if view.moved:
            back = REVERSE[last]
            assert legal[back]
            feasible = bounds[back, ::-1] if view.health < view.fuel else bounds[back]
            allowed = np.all(feasible[np.arange(2), 1 - DELTAS], axis=1)
        else:
            allowed[4] = True
        for i in np.flatnonzero(allowed):
            previous = current - DELTAS[i]
            past, possible = legacy.memory_fields(view, tuple(previous), legal)
            values.append((core, gradient, past, possible, legal))
        assert allowed.any()
    q = net(*legacy.tensor_batch(values)).numpy()
    q[:, ~legal] = -1000
    q -= q[:, legal].max(1, keepdims=True)
    options = np.tile(q[0], (9, 1))
    options[allowed] = q[1:]
    patch, meta = local_image(view)
    # Группы F/R в PrevNet сортируются по запасу. Для прогноза пары оставляем физический порядок.
    order = [1, 0] if view.health < view.fuel else [0, 1]
    signals = np.concatenate((gradient[:, order].reshape(8, 6) / 6,
                              unknown[:, order, 8:].reshape(8, -1)), axis=1)
    x = np.column_stack((core, signals)).astype(np.float32)
    x[:, 0] /= 8
    return dict(x=x, patch=patch, meta=meta, last=np.int64(last),
                stock=np.array([view.fuel, view.health], np.int16), allowed=allowed,
                options=options, missing=q[0], legal=legal)
