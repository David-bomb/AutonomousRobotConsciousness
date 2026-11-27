"""Контроллер принимает только Observation. На входе каждой оценки один разрешённый ход."""
import ctypes as ct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from maze.grid import destination, locations
from maze.interface import Move

MOVES = tuple(Move)
XY = np.asarray([m.value for m in MOVES], np.float32)
LIB = ct.CDLL(str(Path(__file__).with_name("geometry.dll")))
LIB.lengths.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int,
                        np.ctypeslib.ndpointer(np.int32), ct.c_int, np.ctypeslib.ndpointer(np.float32)]
LIB.lengths.restype = None


@lru_cache(maxsize=8192)
def geometry(memory, pos, goal):
    h,w = len(memory), len(memory[0])
    sites = [goal] + locations(memory,"F") + locations(memory,"R")
    assert len(sites)==5  # Формат новой серии: два сервиса каждого типа.
    raw = np.empty((8,19),np.float32)
    ids = np.asarray([y*w+x for x,y in sites],np.int32)
    LIB.lengths("".join(memory).encode("ascii"), w,h,pos[1]*w+pos[0],ids,len(sites),raw)
    return np.asarray(sites, np.float32), raw


def features(v):
    x,y = v.position
    sites,raw = geometry(v.memory, v.position, (x+v.dx,y+v.dy))
    paths = raw[:,:15].reshape(8,5,3)
    legal = np.asarray([destination(v.window,(1,1),m) is not None for m in MOVES])
    # Пять целей: B, F, F, R, R. Расстояния штатных датчиков только из текущей позиции.
    stock = np.array([min(v.fuel,v.health),v.fuel,v.fuel,v.health,v.health],np.float32)
    sensor = np.array([0,v.fuel_distance,v.fuel_distance,v.repair_distance,v.repair_distance],np.float32)
    rel = sites - np.array(v.position,np.float32)
    offset = abs(rel).max(1)
    after = abs(rel[None,:,:] - XY[:,None,:]).max(2)
    columns = []
    def field(array):
        columns.append(np.broadcast_to(array, (8,5)))
    field(offset/60)
    field(offset-after)
    field(stock/100)
    field(sensor/60)
    field(np.clip(stock-sensor,-20,30)/20)
    field(np.clip(stock-offset,-20,30)/20)
    field(abs(sites-sites[0]).max(1)/60)
    for mode in range(3):
        d = paths[:,:,mode]
        best = d.min(0)
        field(np.minimum(d,120)/60)
        field(np.minimum(d-best,16)/4)
        field(np.clip(d-sensor,-24,24)/12)
        field(np.clip(d-stock,-24,24)/12)
        field(d==250)
    targets = np.stack(columns,-1).astype(np.float32)
    last = np.asarray(v.last_move.value if v.last_move else (0,0))
    extras = np.column_stack((np.minimum(raw[:,15],30)/10,np.log1p(raw[:,16])/5,
                              np.minimum(raw[:,17],60)/30,raw[:,18],
                              (XY==last).all(1),(XY==-last).all(1),abs(XY).sum(1)-1,
                              np.full(8,legal.sum()/8), np.full(8,v.fuel/100),np.full(8,v.health/100)))
    # Упорядочиваем виды сервиса по запасу для симметричного переноса F/R.
    if v.health < v.fuel:
        targets = targets[:,[0,3,4,1,2]]
    extras[:,8:] = sorted((v.fuel/100,v.health/100))
    result = np.concatenate((targets.reshape(8,-1),extras),axis=1).astype(np.float32)
    result[~legal]=0
    return result, legal


class Policy(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.site = nn.Sequential(nn.Linear(22,32),nn.ReLU(),nn.Linear(32,24),nn.ReLU())
        self.direction = nn.Sequential(nn.Linear(24*3+10,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.score = nn.Sequential(nn.Linear(width*2,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,x,legal):
        s = self.site(x[:,:,:110].reshape(*x.shape[:2],5,22))
        hidden = self.direction(torch.cat((s[:,:,0], s[:,:,1:3].amax(2), s[:,:,3:5].amax(2), x[:,:,110:]),-1))
        context = hidden.masked_fill(~legal[:,:,None],-1e6).amax(1,keepdim=True).expand_as(hidden)
        return self.score(torch.cat((hidden,context),-1)).squeeze(-1).masked_fill(~legal,-1e6)
