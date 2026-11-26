"""Искать похожие подходы к станциям на точной длине пути в независимых лабиринтах."""
from functools import lru_cache

import numpy as np

from .frontier_reserve import _lengths, _optimistic
from .reference_roads import reference_roads, shifted
from .second_window import ReturnWindows


def matches_on_shell(fields, width, x, y, d, marker_offsets, previous, old):
    # Для каждого похожего окна находим ближайшее смещение маркера среди всех
    # дорог, чья длина в образце в точности равна фактическому замеру.
    anchor, target = np.nonzero(fields[y*width+x, :width*width] == d)
    if previous is not None and old is not None:
        prev = shifted(width,x,y,previous)
        ok = fields[prev[anchor],target] == old
        anchor, target = anchor[ok], target[ok]
    dx, dy = target % width - x[anchor], target // width - y[anchor]
    error = np.minimum.reduce([np.maximum(abs(dx-a),abs(dy-b)) for a,b in marker_offsets])
    best = np.full(len(x), np.inf)
    np.minimum.at(best,anchor,error)
    chosen = error == best[anchor]
    return best, anchor[chosen], target[chosen]


@lru_cache(maxsize=96)
def shell_votes(view, readings, old, prior, sites, options, kind, *, onward=False):
    ox, oy = view.position
    lower = _lengths(_optimistic(view.memory),view.position)
    offsets = [tuple((x-ox,y-oy) for x,y in group if d is not None and lower.get((x,y),np.inf) <= d)
               for group,d in zip(sites,readings)]
    if not offsets[kind]:
        return None
    previous = (prior[0]-ox,prior[1]-oy) if prior is not None else None
    context = [(x-ox,y-oy,c!="#") for y,row in enumerate(view.memory) for x,c in enumerate(row)
               if c != "?" and 1 < max(abs(x-ox),abs(y-oy)) <= 3]
    votes = np.zeros((2,len(options)))
    for draw in range(18):
        road, fields = reference_roads(draw)
        w = len(road)
        y,x = np.nonzero(road)
        for j,row in enumerate(view.window):
            for i,c in enumerate(row):
                match = road[y+j-1,x+i-1] == (c != "#")
                y,x = y[match],x[match]
        if not len(x):
            continue
        error, anchor, target = matches_on_shell(fields,w,x,y,readings[kind],offsets[kind],previous,old[kind])
        if not len(anchor):
            continue
        other = 1-kind
        useful = 1.
        if offsets[other] and readings[other] is not None:
            err,other_anchor,other_targets = matches_on_shell(fields,w,x,y,readings[other],offsets[other],previous,old[other])
            error += err
            budget = (view.fuel,view.health)[other]
            if onward and lower.get((ox+view.dx,oy+view.dy),np.inf) > budget:
                # Даже при заезде на выбранную станцию понадобится пополнение
                # второго ресурса. Оба участка должны быть в одной аналогии.
                limit = min(99,budget-readings[kind])
                useful = connected_stops(fields,anchor,target,other_anchor,other_targets,limit)
                for bx,by in sites[other]:
                    b = shifted(w,x,y,(bx-ox,by-oy))
                    compatible = fields[y*w+x,b] >= readings[other]
                    if previous is not None and old[other] is not None:
                        compatible &= fields[shifted(w,x,y,previous),b] >= old[other]
                    useful |= compatible[anchor] & (fields[target,b[anchor]] <= limit)
        # Одна клетка смещения в образце вдвое уменьшает вес аналогии.
        # Совпадение дальней известной окрестности сравнивается отдельно.
        mismatches = np.zeros(len(x), dtype=int)
        for dx,dy,free in context:
            p = shifted(w,x,y,(dx,dy))
            mismatches += (fields[p,p] == 0) != free
        weight = np.exp2(-error.astype(float))/(1.+mismatches)**2
        ties = np.bincount(anchor,minlength=len(x))
        slopes = np.array([fields[shifted(w,x,y,m.value)[anchor],target] == readings[kind]-1 for m in options])
        votes[draw%2] += (slopes * (weight[anchor]/ties[anchor]/np.maximum(1,slopes.sum(axis=0))) * useful).sum(axis=1)
    return votes


def connected_stops(fields, anchor, target, other_anchor, other_targets, limit):
    start = np.searchsorted(other_anchor,anchor)
    count = np.searchsorted(other_anchor,anchor,side="right")-start
    fits = np.zeros(len(anchor),dtype=bool)
    for i in range(int(count.max(initial=0))):
        idx = np.flatnonzero(count>i)
        fits[idx] |= fields[target[idx],other_targets[start[idx]+i]] <= limit
    return fits


class ReferenceShells(ReturnWindows):
    require_agreement = False
    onward = False

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        own, other = (view.fuel,view.health) if kind == 0 else (view.health,view.fuel)
        if readings[kind] is None or not 0 <= min(own,other-1)-readings[kind] < 2:
            return parent
        moves = tuple(options)
        votes = shell_votes(view, readings, old, prior, sites, moves, kind,onward=self.onward)
        if votes is None or not votes.sum(axis=1).all():
            return parent
        total = votes.sum(axis=0)
        best = max(range(len(moves)), key=lambda i:(total[i],moves[i] == parent))
        if self.require_agreement and not (votes[:,best] > votes[:,moves.index(parent)]).all():
            return parent
        return moves[best]


class ConsistentShells(ReferenceShells):
    require_agreement = True
