"""Самостоятельная замороженная сеть. Единственный аргумент choose — Observation."""
import hashlib
from io import BytesIO
from pathlib import Path

import torch
from torch import nn
from maze.interface import Move, Observation
from .features import MOVES, encode


class Explorer(nn.Module):
    def __init__(self, edge_width, token_width):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(edge_width,64),nn.ReLU(),nn.Linear(64,48),nn.ReLU())
        self.front=nn.Sequential(nn.Linear(token_width,32),nn.ReLU(),nn.Linear(32,24),nn.ReLU())
        self.branch=nn.Sequential(nn.Linear(48+2*24,64),nn.ReLU(),nn.Linear(64,48),nn.ReLU())
        self.score=nn.Sequential(nn.Linear(96,48),nn.ReLU(),nn.Linear(48,1))

    def forward(self, edges, tokens, legal):
        mask=tokens[:,:,:,0:1]>0
        front=self.front(tokens)*mask
        maximum=front.amax(2)
        average=front.sum(2)/mask.sum(2).clamp_min(1)
        branch=self.branch(torch.cat((self.edge(edges),maximum,average),dim=-1))
        context=branch.masked_fill(~legal[:,:,None],-1e6).amax(1,keepdim=True).expand_as(branch)
        return self.score(torch.cat((branch,context),-1)).squeeze(-1).masked_fill(~legal,-1e6)


class Controller:
    def __init__(self, weights, expected_sha256):
        data=Path(weights).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected_sha256:
            raise ValueError("Изменились веса модели")
        self._load(data)

    def _load(self,data):
        state=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
        self.model=Explorer(state["edge.0.weight"].shape[1],state["front.0.weight"].shape[1])
        self.model.load_state_dict(state)
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def choose(self, view: Observation) -> Move:
        edges,tokens,legal=encode(view)
        scores=self.model(torch.from_numpy(edges[None]).float(),torch.from_numpy(tokens[None]).float(),
                          torch.from_numpy(legal[None]))
        return MOVES[int(scores.argmax(1).item())]


def from_bytes(data):
    model=Controller.__new__(Controller)
    model._load(data)
    return model
