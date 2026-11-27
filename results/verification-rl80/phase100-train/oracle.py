"""Послойный точный поиск по физическим состояниям, только для внешнего учителя train."""
from collections import deque
from maze.grid import destination, locations, wave
from maze.interface import Move


class Exact:
    def __init__(self, level):
        self.goal = level.goal
        self.dist = {p:d for p,d,_ in wave(level.grid,[self.goal])}
        self.stations = {p:s for s in "FR" for p in locations(level.grid,s)}
        self.near = [{p:d for p,d,_ in wave(level.grid,locations(level.grid,s))} for s in "FR"]
        self.edges = {p:[(i,q) for i,m in enumerate(Move) if (q:=destination(level.grid,p,m)) is not None]
                      for p in self.dist}

    def route(self, env):
        remaining = env.limit-env.steps
        todo = deque([(env.position,env.fuel,env.health,b"")])
        labels = {}
        while todo:
            p,f,h,path = todo.popleft()
            if p == self.goal:
                return path
            d = self.dist[p]
            if min(f,h)<=0 or len(path)+d>remaining:
                continue
            if min(d,self.near[0][p])>f or min(d,self.near[1][p])>h:
                continue
            seen = labels.setdefault(p,[])
            if any(old_f>=f and old_h>=h for old_f,old_h in seen):
                continue
            seen.append((f,h))
            for i,q in self.edges[p]:
                k = self.stations.get(q)
                todo.append((q,100 if k=="F" else f-1,100 if k=="R" else h-1,path+bytes([i])))
        return None
