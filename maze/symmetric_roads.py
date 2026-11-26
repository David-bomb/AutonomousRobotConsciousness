"""Одни и те же гипотезы о дорогах во всех ориентациях локальной системы координат."""
from functools import lru_cache

import numpy as np

from .interface import Move
from .pattern_roads import MarkedStreets, pattern_votes


@lru_cache(maxsize=128)
def oriented_grids(memory):
    board=np.asarray([list(row) for row in memory])
    h,w=board.shape
    frames=[]
    for reflected in (False,True):
        grid=board[:,::-1] if reflected else board
        point=(w-1,0) if reflected else (0,0)
        east=(-1,0) if reflected else (1,0)
        south=(0,1)
        for _ in range(4):
            frames.append((tuple("".join(row) for row in grid),point,east,south))
            point=(grid.shape[0]-1-point[1],point[0])
            east=(-east[1],east[0])
            south=(-south[1],south[0])
            grid=np.rot90(grid,-1)
    return tuple(frames)


class SymmetricStreets(MarkedStreets):
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        goal=x+view.dx,y+view.dy
        total=[{},{}]
        for grid,origin,east,south in oriented_grids(view.memory):
            def transform(p):
                a,b=p
                return origin[0]+a*east[0]+b*south[0], origin[1]+a*east[1]+b*south[1]
            groups=tuple(tuple(transform(p) for p in row) for row in sites)
            scores=pattern_votes(grid,transform(view.position),readings,old,
                                 transform(prior) if prior is not None else None,
                                 groups,transform(goal),32)
            for move in Move:
                dx,dy=move.value
                rotated=Move((dx*east[0]+dy*south[0],dx*east[1]+dy*south[1]))
                for out,score in zip(total,scores):
                    if rotated in score:
                        out[move]=out.get(move,0.)+score[rotated]
        return total
