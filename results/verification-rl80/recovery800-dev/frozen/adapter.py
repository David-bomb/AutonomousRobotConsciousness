"""Изолированный интерфейс замороженных весов: Observation -> Move."""
from hashlib import sha256
from pathlib import Path

import torch
from maze.interface import Observation, Move
from .policy import MOVES, Policy, features


class Controller:
    def __init__(self, weights, expected_sha256):
        path=Path(weights)
        if sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError("Контрольная сумма весов не совпала")
        torch.set_num_threads(1)
        self.model=Policy().eval()
        self.model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def choose(self, view: Observation) -> Move:
        inputs,legal=features(view)
        scores=self.model(torch.from_numpy(inputs[None]),torch.from_numpy(legal[None]))
        return MOVES[int(scores.argmax())]
