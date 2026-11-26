"""Оценка первого хода по дорогам ровно измеренной длины."""
from functools import lru_cache

import numpy as np

from .frontier_reserve import _lengths
from .interface import Move
from .passage_automaton import Passage
from .resource_automaton import possible_directions


@lru_cache(maxsize=48)
def transitions(memory, confirmed_diagonals=False, turn_weights=None):
    height, width = len(memory), len(memory[0])
    board = np.pad(np.array([list(row) for row in memory]),1,constant_values="#").ravel()
    stride = width+2
    points = np.arange(len(board)).reshape(height+2,stride)[1:-1,1:-1].ravel()
    offsets = [dx+dy*stride for dx,dy in (m.value for m in Move)]
    count = len(points)
    indices = np.full(len(board),count)
    indices[points] = np.arange(count)
    neighbors = np.array([indices[points+offset] for offset in offsets])
    seen = board[points] != "?"
    p_open = (1+np.count_nonzero(seen & (board[points]!="#")))/(2+np.count_nonzero(seen))
    weights = []
    for (dx,dy),offset in zip((m.value for m in Move),offsets):
        dependencies = [board[points+offset]]
        if dx and dy:
            dependencies.extend([board[points+dx],board[points+dy*stride]])
        valid = (board[points]!="#") & np.all(np.array(dependencies)!="#",axis=0)
        if confirmed_diagonals and dx and dy:
            valid &= (board[points]!="?") & np.all(np.array(dependencies)!="?",axis=0)
        weights.append(valid * p_open**np.sum(np.array(dependencies)=="?",axis=0))
    weights = np.array(weights)
    turns = []
    for i,(ax,ay) in enumerate(m.value for m in Move):
        row = weights.copy()
        for j,(bx,by) in enumerate(m.value for m in Move):
            dx,dy = ax+bx,ay+by
            if max(abs(dx),abs(dy)) <= 1:
                # Возврат либо путь, который уже можно сократить одним ходом.
                if dx==0 or dy==0:
                    row[j] = 0.
                else:
                    # Для двух перпендикулярных ходов достаточно проверить
                    # четвёртый угол квадрата, три остальные клетки лежат на пути.
                    corner = board[points-offsets[i]+offsets[j]]
                    row[j] *= (corner=="#") | (corner=="?")
            elif i==j and turn_weights is None:
                row[j] *= 2
            if turn_weights is not None:
                row[j] *= turn_weights[i][j]
        total = row.sum(axis=0)
        turns.append(row/np.where(total>0,total,1.))
    return neighbors,np.array(turns)


def upper(memory, sites):
    width = len(memory[0])
    bound = np.full(width*len(memory),np.inf)
    for site in sites:
        for (x,y),d in _lengths(memory,site).items():
            i = y*width+x
            bound[i] = min(bound[i],d)
    return bound


def suffixes(view,service,old,confirmed_diagonals=False,turn_weights=None):
    grid = view.memory
    width = len(grid[0])
    measured = service.distance
    neighbors,turns = transitions(grid,confirmed_diagonals,turn_weights)
    from_start = upper(grid,[view.position])
    to_sites = upper(grid,service.sites)
    prior_bound = np.full_like(from_start,np.inf)
    if old is not None and view.moved:
        dx,dy = view.last_move.value
        prior = view.position[0]-dx,view.position[1]-dy
        prior_bound = upper(grid,[prior]) + measured-old
    # Нулевые хвосты заканчиваются на сервисах. Дальше считаем допустимые
    # продолжения в обратном порядке, оставляя первый ход неизвестным.
    values = np.zeros((len(Move),len(from_start)+1))
    values[:,:-1] = to_sites==0
    yield measured,values
    indices = np.arange(len(Move))[:,None]
    for t in range(measured-1,0,-1):
        successors = values[indices,neighbors]
        feasible = (t<=from_start) & (measured-t<=to_sites) & (t<=prior_bound)
        values[:,:-1] = np.einsum("ijn,jn->in",turns,successors)*feasible
        yield t,values


def bridge_evidence(view,service,old,confirmed_diagonals=False):
    grid = view.memory
    options = possible_directions(view,service,old)
    if not options or service.distance<=0:
        return {}
    origin = view.position[1]*len(grid[0])+view.position[0]
    neighbors,_ = transitions(grid,confirmed_diagonals)
    for _,values in suffixes(view,service,old,confirmed_diagonals):
        pass
    scores = {m:values[i,neighbors[i,origin]] for i,m in enumerate(Move) if m in options}
    best = max(scores.values())
    return {m:v/best for m,v in scores.items()} if best else {}


class BridgePassage(Passage):
    service_evidence = staticmethod(bridge_evidence)


class CorridorBridge(BridgePassage):
    @staticmethod
    def service_evidence(view,service,old):
        # За пределами окна сначала проверяем объяснение замера без
        # воображаемого свободного квадрата. Реальные диагонали доступны всегда.
        return bridge_evidence(view,service,old,True) or bridge_evidence(view,service,old)
