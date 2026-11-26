"""Предсказывать продолжение неизвестного окна по полностью раскрытым окнам этой поездки."""
from collections import Counter
from functools import lru_cache, partial

from .frontier_automaton import _frontiers
from .grid import cell
from .interface import Move
from .passage_flow import passage_graph
from .second_window import ReturnWindows, closed_probability
from .simple_exit_paths import simple_exit_probabilities


OFFSETS = tuple(m.value for m in Move)
BITS = {offset:1<<i for i,offset in enumerate(OFFSETS)}
NEEDS = tuple(BITS[(dx,dy)] | ((BITS[(dx,0)] | BITS[(0,dy)]) if dx and dy else 0) for dx,dy in OFFSETS)


@lru_cache(maxsize=256)
def orientations(bits):
    points = {offset for offset,bit in BITS.items() if bits & bit}
    values = []
    for _ in range(4):
        values.append(sum(BITS[(x,y)] for x,y in points))
        values.append(sum(BITS[(-x,y)] for x,y in points))
        points = {(-y,x) for x,y in points}
    return values


def observation(memory, point, goal):
    x,y = point
    known = opened = 0
    for offset,bit in BITS.items():
        dx,dy = offset
        q = x+dx,y+dy
        kind = cell(memory,*q)
        if kind != "?" or q == goal:
            known |= bit
            if kind != "#":
                opened |= bit
    return known,opened


@lru_cache(maxsize=32)
def known_patterns(memory):
    counts = Counter()
    for y,row in enumerate(memory):
        for x,c in enumerate(row):
            if c in "#?":
                continue
            known,opened = observation(memory,(x,y),None)
            if known == 255:
                counts.update(orientations(opened))
    return counts


@lru_cache(maxsize=1024)
def forecast(memory, point, goal):
    visible,roads = observation(memory,point,goal)
    uncertain = [(i,need) for i,need in enumerate(NEEDS) if need & visible != need and not (need & visible & ~roads)]
    edges = [0.] * len(OFFSETS)
    closed = total = 0.
    for bits,count in known_patterns(memory).items():
        if bits & visible != roads:
            continue
        weight = count/8
        total += weight
        possible = False
        for i,need in uncertain:
            if bits & need == need:
                edges[i] += weight
                possible = True
        if not possible:
            closed += weight
    return total,closed,tuple(edges)


def predicted_closure(memory, point, goal):
    dx,dy = goal[0]-point[0],goal[1]-point[1]
    if (dx,dy) in BITS and cell(memory,*goal)=="?":
        # Обязательная проходимость цели уже входит в точный перебор исходного правила.
        return closed_probability(memory,point,goal)
    total,closed,_ = forecast(memory,point,goal)
    return (closed+closed_probability(memory,point,goal))/(1+total)


@lru_cache(maxsize=32)
def patterned_graph(memory, goal):
    graph = passage_graph(memory,goal).copy()
    width = len(memory[0])
    changes = {}
    for p in _frontiers(memory):
        total,_,edges = forecast(memory,p,goal)
        if not total:
            continue
        visible,roads = observation(memory,p,goal)
        u = p[1]*width+p[0]
        for i,need in enumerate(NEEDS):
            if need & visible == need or need & visible & ~roads:
                continue
            dx,dy = OFFSETS[i]
            v = (p[1]+dy)*width+p[0]+dx
            key = (min(u,v),max(u,v))
            changes.setdefault(key,[]).append((edges[i],total))
    for (u,v),samples in changes.items():
        old = graph[u,v]
        if old:
            chance = (old + sum(n for n,_ in samples))/(1+sum(d for _,d in samples))
            graph[u,v] = graph[v,u] = chance
    return graph


class WindowPatterns(ReturnWindows):
    closed_chance = staticmethod(predicted_closure)


class PassagePatterns(WindowPatterns):
    probabilities = staticmethod(partial(simple_exit_probabilities,graph_builder=patterned_graph))
