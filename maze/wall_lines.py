"""Оценка продолжения видимых прямых стен при сравнении дорог измеренной длины."""
from functools import lru_cache
from math import floor,log

from .street_trunk import StreetTrunk,alternatives


@lru_cache(maxsize=128)
def line_patterns(memory,goal):
    h,w=len(memory),len(memory[0])
    proposed=set()
    for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
        continues=stops=0
        ends=[]
        for y,row in enumerate(memory):
            for x,kind in enumerate(row):
                if kind!="#":
                    continue
                bx,by=x-dx,y-dy
                nx,ny=x+dx,y+dy
                if not (0<=bx<w and 0<=by<h and 0<=nx<w and 0<=ny<h) or memory[by][bx]!="#":
                    continue
                next_kind=memory[ny][nx]
                if next_kind=="?":
                    ends.append((nx,ny))
                elif next_kind=="#":
                    continues+=1
                else:
                    stops+=1
        probability=(1+continues)/(2+continues+stops)
        depth=floor(log(.5)/log(probability))
        for x,y in ends:
            for _ in range(depth):
                if not (0<=x<w and 0<=y<h) or memory[y][x]!="?" or (x,y)==goal:
                    break
                proposed.add((x,y))
                x,y=x+dx,y+dy
    grids=alternatives(memory,goal)
    if not proposed:
        return grids
    weights={}
    for weight,grid in grids:
        walls=[list(row) for row in grid]
        for x,y in proposed:
            walls[y][x]="#"
        extended=tuple("".join(row) for row in walls)
        for candidate in (grid,extended):
            weights[candidate]=weights.get(candidate,0.)+weight/2
    return tuple((weight,grid) for grid,weight in weights.items())


class WallLines(StreetTrunk):
    road_patterns=staticmethod(line_patterns)
