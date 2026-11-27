"""Восемь представлений того же текущего наблюдения с обратным соответствием ходов."""
from dataclasses import replace
from maze.interface import Move

MOVES=tuple(Move)


def vector(x,y,index):
    for _ in range(index%4):
        x,y=y,-x
    return (-x if index>=4 else x),y


def raster(grid,index):
    for _ in range(index%4):
        grid=tuple("".join(row) for row in zip(*grid))[::-1]
    return tuple(row[::-1] for row in grid) if index>=4 else grid


def observation(v,index):
    if index==0:return v
    w,h=len(v.memory[0]),len(v.memory)
    x,y=v.position
    for _ in range(index%4):
        x,y=y,w-1-x
        w,h=h,w
    if index>=4:x=w-1-x
    dx,dy=vector(v.dx,v.dy,index)
    last=Move(vector(*v.last_move.value,index)) if v.last_move else None
    return replace(v,memory=raster(v.memory,index),window=raster(v.window,index),position=(x,y),
                   dx=dx,dy=dy,last_move=last)


ORDER=tuple(tuple(MOVES.index(Move(vector(*a.value,j))) for a in MOVES) for j in range(8))
