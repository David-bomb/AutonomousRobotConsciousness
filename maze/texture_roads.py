"""Проверяет, продолжают ли неизвестные клетки локальный рисунок открытых дорог."""
from functools import lru_cache, partial

import numpy as np
from scipy.special import expit, logit

from .cell_hypotheses import CellRoads, cell_votes


@lru_cache(maxsize=128)
def neighborhood_laws(memory):
    h,w=len(memory),len(memory[0])
    observed=np.frombuffer("".join(memory).encode(),dtype="S1").reshape(h,w)
    # Отдельный известный маркер сервиса не сообщает типы соседних клеток.
    observed=np.where(observed==b"?",-1,(observed!=b"#").astype(int))
    counts=np.ones(16)
    for turn in range(4):
        for patch in (np.rot90(observed,turn),np.rot90(observed.T,turn)):
            nw,ne,sw,se=patch[:-1,:-1],patch[:-1,1:],patch[1:,:-1],patch[1:,1:]
            valid=(nw>=0)&(ne>=0)&(sw>=0)&(se>=0)
            ix=(nw+2*ne+4*sw+8*se)[valid]
            # Восемь симметрий представляют один физический квадрат.
            counts+=np.bincount(ix,minlength=16)/8
    return np.log(counts[1::2]/counts[::2])


def texture_rank(memory,board,prob,rng):
    h,w=len(memory),len(memory[0])
    law=neighborhood_laws(memory)
    field=(rng.random((h,w))<prob.reshape(h,w)).astype(np.uint8)
    fixed=(board!=b"?").reshape(h,w)
    free=(board!=b"#").reshape(h,w)
    field[fixed]=free[fixed]
    # Обновление по четырём непересекающимся классам квадратной решётки.
    # За границей поля состояние неизвестно. Краевые квадраты пропускаются.
    bias=logit(prob.reshape(h,w))
    def forecast():
        local=np.zeros((h,w))
        count=np.zeros((h,w))
        for dy in (-1,1):
            for dx in (-1,1):
                y=slice(0,h-1) if dy>0 else slice(1,h)
                x=slice(0,w-1) if dx>0 else slice(1,w)
                cy=slice(1,h) if dy>0 else slice(0,h-1)
                cx=slice(1,w) if dx>0 else slice(0,w-1)
                local[y,x]+=law[field[y,cx]+2*field[cy,x]+4*field[cy,cx]]
                count[y,x]+=1
        return expit((bias+local/np.maximum(count,1))/2)
    for _ in range(2):
        for color in rng.permutation(4):
            region=np.s_[color//2::2,color%2::2]
            p=forecast()[region]
            field[region]=np.where(fixed[region],free[region],rng.random(p.shape)<p)
    p=forecast().ravel()
    u=rng.random(len(board))
    return (u/(1-u))*((1-p)/p)


class TextureCells(CellRoads):
    vote_fields=staticmethod(partial(cell_votes,ranker=texture_rank))
