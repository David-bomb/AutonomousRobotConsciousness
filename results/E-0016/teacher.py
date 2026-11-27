"""Разметка только учебных геометрий, изолирована от оценочной политики."""
from functools import lru_cache
import numpy as np
from maze.grid import locations, wave, destination
from maze.interface import Move

MOVES = tuple(Move)
INF = 100000


class Teacher:
    def __init__(self,level):
        self.sites = locations(level.grid,"F") + locations(level.grid,"R")
        self.dist = [{p:d for p,d,_ in wave(level.grid,[s])} for s in [*self.sites,level.goal]]
        self.level = level
        self.remaining = lru_cache(maxsize=65536)(self._remaining)

    def _remaining(self,p,f,h,legs):
        direct = self.dist[-1][p]
        if direct<=min(f,h):
            return direct
        best=INF
        if legs and min(f,h)>0:
            for i,s in enumerate(self.sites):
                d = self.dist[i][p]
                if not d or d>=best:
                    continue
                nf,nh = (100,h-d) if i<2 else (f-d,100)
                if d<=min(f,h) and min(nf,nh)>0:
                    best=min(best,d+self.remaining(s,nf,nh,legs-1))
        return best

    def labels(self,env):
        scores = np.full(8,INF)
        for a,m in enumerate(MOVES):
            q=destination(self.level.grid,env.position,m)
            if q is None:
                continue
            tile=self.level.grid[q[1]][q[0]]
            f=100 if tile=="F" else env.fuel-1
            h=100 if tile=="R" else env.health-1
            scores[a]=self.remaining(q,f,h,5)
        viable=scores<min(INF,env.limit-env.steps)
        best=(scores==scores.min()) & viable
        return best,viable
