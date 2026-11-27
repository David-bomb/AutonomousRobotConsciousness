"""Дополнительное чтение уже доступной карты в контроллере Double DQN."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from neural_rl.networks import Network


class QueryNetwork(Network):
    def __init__(self, config, side):
        super().__init__(config)
        if self.family != "double":
            raise ValueError("E-0015 использует Double DQN")
        self.side = side
        self.heads = 4
        self.cell_dims = 11  # Семь исходных битов, dx, dy, |dx|, |dy|.
        self.query = nn.Linear(29 + 9*7, self.heads*self.cell_dims)
        self.combine = nn.Linear(self.heads*self.cell_dims, self.shared[0].in_features-29, bias=False)
        nn.init.zeros_(self.combine.weight)
        axis = torch.arange(side, dtype=torch.float32) / side
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("xy", torch.stack((x,y)).reshape(1,2,-1), persistent=False)
        offsets = [dy*(side+2)+dx for dy in (-1,0,1) for dx in (-1,0,1)]
        self.register_buffer("window_offsets",torch.tensor(offsets),persistent=False)

    def read_cells(self, image, planes, numbers):
        x,y = (numbers[:,:2]*self.side).round().long().unbind(1)
        centers = (y+1)*(self.side+2)+x+1
        padded = F.pad(image,(1,1,1,1),value=2).flatten(1)
        window = padded.gather(1,centers[:,None]+self.window_offsets)
        window_bits = (window[:,None,:].bitwise_and(self.bits.reshape(1,7,1))!=0).float()
        query = self.query(torch.cat((numbers,window_bits.flatten(1)),dim=1))
        query = query.reshape(-1,self.heads,self.cell_dims)
        relative = self.xy-numbers[:,:2,None]
        cells = torch.cat((planes.flatten(2),relative,relative.abs()),dim=1)
        # Стена, дорога, сервис, цель или сам робот. Неизвестные клетки
        # и дополнение за краем карты остаются доступны основной CNN.
        visible = image.flatten(1).bitwise_and(126)!=0
        logits = torch.bmm(query,cells) / math.sqrt(self.cell_dims)
        weights = logits.masked_fill(~visible[:,None,:],-torch.inf).softmax(dim=2)
        return torch.bmm(weights,cells.transpose(1,2)).flatten(1)

    def forward(self, data):
        image,numbers,legal = data
        planes = (image[:,None].bitwise_and(self.bits)!=0).float()
        spatial = self.map(planes) + self.combine(self.read_cells(image,planes,numbers))
        hidden = self.shared(torch.cat((spatial,numbers),dim=1))
        q = self.advantage(hidden)
        return q.masked_fill(~legal,-1e9), q.unsqueeze(-1), None
