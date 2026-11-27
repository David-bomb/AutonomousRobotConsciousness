"""Изолированный интерфейс замороженных весов: Observation -> Move."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import torch
from maze.interface import Observation, Move
from .policy import MOVES, Policy, features
from .critical_policy import CriticalPolicy
from .rich_policy import RichPolicy, inputs


class Controller:
    def __init__(self, weights, expected_sha256):
        path=Path(weights)
        data=path.read_bytes()
        if sha256(data).hexdigest()!=expected_sha256:
            raise ValueError("Контрольная сумма весов не совпала")
        self._load(data)

    def _load(self, data):
        torch.set_num_threads(1)
        state=torch.load(BytesIO(data),map_location="cpu",weights_only=True)
        self.prior=None
        if "wall_prior" in state:
            self.model=RichPolicy().eval()
            self.prior=state["wall_prior"].numpy().tobytes()
        else:
            self.model=(CriticalPolicy() if "risk.0.weight" in state else Policy()).eval()
        self.model.load_state_dict(state)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def choose(self, view: Observation) -> Move:
        x,legal=features(view) if self.prior is None else inputs(view,self.prior)
        scores=self.model(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))
        return MOVES[int(scores.argmax())]


def from_bytes(data):
    controller=Controller.__new__(Controller)
    controller._load(data)
    return controller
