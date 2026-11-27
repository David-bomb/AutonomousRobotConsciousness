"""Свёрточный аппарат принятия решений и целевые распределения Rainbow."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def batch(states):
    return tuple(torch.from_numpy(np.stack(column)).to("cuda") for column in zip(*states))


class NoisyLinear(nn.Module):
    """Линейный слой с обучаемой амплитудой факторизованного шума."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.base = nn.Linear(inputs, outputs)
        self.weight_sigma = nn.Parameter(torch.full((outputs, inputs), 0.5 / math.sqrt(inputs)))
        self.bias_sigma = nn.Parameter(torch.full((outputs,), 0.5 / math.sqrt(outputs)))
        self.register_buffer("incoming", torch.zeros(inputs), persistent=False)
        self.register_buffer("outgoing", torch.zeros(outputs), persistent=False)

    def resample(self):
        for buffer in (self.incoming, self.outgoing):
            normal = torch.randn_like(buffer)
            buffer.copy_(normal.sign() * normal.abs().sqrt())

    def forward(self, x):
        if not self.training:
            return self.base(x)
        weight = self.base.weight + self.weight_sigma * torch.outer(self.outgoing, self.incoming)
        return F.linear(x, weight, self.base.bias + self.bias_sigma * self.outgoing)


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.family = family = config["family"]
        channels, hidden, depth = config["architecture"]
        self.actions = 4 if family == "hrl_astar" else 8
        self.atoms = 51 if family == "rainbow" else 1
        self.register_buffer("support", torch.linspace(-12, 12, self.atoms))
        self.register_buffer("bits", (2**torch.arange(7, dtype=torch.uint8)).view(1, 7, 1, 1))
        layers = [nn.Conv2d(7, channels, 5, stride=3, padding=2), nn.ReLU()]
        if depth == 2:
            layers.extend([nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU()])
        self.map = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten())
        self.shared = nn.Sequential(nn.Linear(channels*9 + 29, hidden), nn.ReLU())
        linear = NoisyLinear if family == "rainbow" else nn.Linear
        def head(outputs):
            return nn.Sequential(linear(hidden, hidden), nn.ReLU(), linear(hidden, outputs))
        self.advantage = head(self.actions * self.atoms)
        self.value = head(self.atoms) if family in ("dueling", "rainbow", "ppo") else None

    def noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.resample()

    def forward(self, data):
        image, numbers, mask = data
        planes = (image.unsqueeze(1).bitwise_and(self.bits) != 0).float()
        features = self.shared(torch.cat([self.map(planes), numbers], dim=1))
        advantage = self.advantage(features).reshape(-1, self.actions, self.atoms)
        value = self.value(features) if self.value is not None else None
        if self.family in ("dueling", "rainbow"):
            advantage = value.unsqueeze(1) + advantage - advantage.mean(1, keepdim=True)
        scores = ((advantage.softmax(-1) * self.support).sum(-1) if self.atoms > 1
                  else advantage.squeeze(-1))
        return scores.masked_fill(~mask, -1e9), advantage, value


def project(probabilities, reward, discount, support):
    targets = reward[:, None] + discount[:, None] * support
    positions = ((targets-support[0]) / (support[-1]-support[0]) * (len(support)-1)).clamp(0, len(support)-1)
    lower = positions.long()
    upper = (lower + 1).clamp_max(len(support)-1)
    fraction = positions - lower
    result = torch.zeros_like(probabilities)
    result.scatter_add_(1, lower, probabilities*(1-fraction))
    result.scatter_add_(1, upper, probabilities*fraction)
    return result
