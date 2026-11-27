"""Обучаемые оценки отдельных выходов на неоткрытые клетки и восьми физических ходов."""
import torch
from torch import nn


class GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.SiLU(),
                                  nn.Linear(64, 48), nn.SiLU())
        self.candidate = nn.Sequential(nn.Linear(48*3, 64), nn.SiLU(), nn.Linear(64, 1))
        self.action = nn.Sequential(nn.Linear(48*3+3, 64), nn.LayerNorm(64), nn.SiLU(),
                                    nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, x):
        exists = x[...,0] > 0
        h = self.edge(x)
        per_move = h.masked_fill(~exists[...,None], -1e3).amax(2)
        per_move = torch.where(exists.any(2)[...,None], per_move, 0.)
        scene = per_move.amax(1,keepdim=True).expand_as(per_move)
        q = self.candidate(torch.cat((h, per_move.unsqueeze(2).expand_as(h),
                                      scene.unsqueeze(2).expand_as(h)), -1)).squeeze(-1)
        odds = q.sigmoid().masked_fill(~exists, 0.)
        pooled = (h*odds[...,None]).sum(2)/(odds.sum(2,keepdim=True)+.01)
        evidence = torch.stack((odds.amax(2), odds.sum(2).log1p(), exists.float().sum(2)/8), -1)
        return q, self.action(torch.cat((per_move, scene, pooled, evidence), -1))
