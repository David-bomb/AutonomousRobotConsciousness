"""Адаптер восьми оценок Policy из пилота E-0016. На входе только наблюдение."""
from importlib import import_module
from io import BytesIO

import torch

from policy_api import GreedyController


def load(checkpoint: bytes):
    policy = import_module("results.E-0016.policy")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = policy.Policy().cpu().eval()
    net.load_state_dict(torch.load(BytesIO(checkpoint), map_location="cpu", weights_only=True),
                        strict=True)

    @torch.inference_mode()
    def scores(view):
        x, legal = policy.features(view)
        mask = torch.from_numpy(legal).unsqueeze(0)
        q = net(torch.from_numpy(x).unsqueeze(0), mask)
        return q.masked_fill(~mask, -torch.inf)[0].tolist()

    return GreedyController(scores)
