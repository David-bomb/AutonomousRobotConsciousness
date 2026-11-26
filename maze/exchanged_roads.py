"""Искать сети с измеренными длинами перестановкой проходов без разрыва связности."""
from collections import Counter, deque
from functools import lru_cache
from math import exp

import numpy as np

from .compatible_loops import feasible_moves, measured_constraints
from .exact_worlds import layers, pack
from .frontier_reserve import _lengths, _optimistic
from .grid import locations
from .pattern_roads import cells_from_parts, patterns, road_parts, scaffold
from .route_witness import geometry
from .standby_turn import RankedStandby


def cycle_path(start, end, neighbors, used):
    previous = {start:None}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if p == end:
            break
        for q,edge in neighbors[p]:
            if used[edge] and q not in previous:
                previous[q] = p,edge
                queue.append(q)
    path = []
    p = end
    while p != start:
        p,edge = previous[p]
        path.append(edge)
    return path


def mismatch(mask, shifts, constraints):
    error = 0
    for markers,bounds in constraints:
        radius = max(d for _,d in bounds)+8
        reach = layers(mask,shifts,markers,radius)
        for mark,d in bounds:
            distance = next((t for t,layer in enumerate(reach) if mark & layer),radius+1)
            error += abs(distance-d)
    return error


@lru_cache(maxsize=64)
def exchanged_samples(memory,goal,origin,gauges,prior,old):
    w,h = len(memory[0]),len(memory)
    shifts,_ = geometry(w,h)
    constraints = measured_constraints(memory,origin,gauges,prior,old)
    optimistic = _optimistic(memory)
    possible = set()
    for k,d in zip("FR",gauges):
        if d is None:
            continue
        for marker in locations(memory,k):
            fields = _lengths(optimistic,marker)
            for p,leg in _lengths(optimistic,origin).items():
                if leg+fields.get(p,1000)<=d:
                    possible.add(p[1]*w+p[0])
    board,models,weights = patterns(memory,goal)
    mutable = pack(board==b"?") & ~(1 << (goal[1]*w+goal[0]))
    rng = np.random.default_rng(42)
    accepted = []
    for trial in range(12):
        model = models[rng.choice(len(models),p=weights)]
        if model is None:
            continue
        _,left,right,doors,_,_,_ = scaffold(memory,model)
        tree,extras = road_parts(memory,model,rng,depth=bool(trial%2))
        used = tree | extras
        mask = pack(cells_from_parts(memory,model,board,used,goal[1]*w+goal[0]))
        parts = [sum(1<<int(p) for p in door) & mutable for door in doors]
        neighbors = [[] for _ in range(w*h)]
        nearby = []
        for e in rng.permutation(len(left)).tolist():
            a,b = int(left[e]),int(right[e])
            neighbors[a].append((b,e))
            neighbors[b].append((a,e))
            if parts[e] and (a in possible or b in possible or any(p in possible for p in doors[e])):
                nearby.append(e)
        error = mismatch(mask,shifts,constraints)
        matches = {mask} if error==0 else set()
        for _ in range(48):
            closed = [e for e in nearby if not used[e]]
            if not closed:
                break
            add = closed[int(rng.integers(len(closed)))]
            path = cycle_path(int(left[add]),int(right[add]),neighbors,used)
            removable = [e for e in path if parts[e]]
            if not removable:
                continue
            remove = removable[int(rng.integers(len(removable)))]
            proposal = (mask | parts[add]) & ~parts[remove]
            changed = mismatch(proposal,shifts,constraints)
            if changed==0:
                matches.add(proposal)
            if changed <= error or rng.random() < exp(error-changed):
                used[add],used[remove] = True,False
                mask,error = proposal,changed
        if matches:
            accepted.append(tuple(sorted(matches)))
    return tuple(accepted)


class ExchangedRoads(RankedStandby):
    preserve = False
    sample_trials = staticmethod(exchanged_samples)

    def trial_weights(self, view, trials):
        return {}

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        parent = super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        stocks = view.fuel,view.health
        if readings[kind] is None or not 0<=min(stocks[kind],stocks[1-kind]-1)-readings[kind]<2:
            return parent
        goal = view.position[0]+view.dx,view.position[1]+view.dy
        trials = self.sample_trials(view.memory,goal,view.position,readings,prior,old)
        weights = self.trial_weights(view,trials)
        totals = Counter()
        preserved = set(options)
        usable = 0
        for trial in trials:
            votes = Counter()
            for mask in trial:
                safe = feasible_moves(view,goal,mask,options)
                votes.update({m:weights.get(mask,1.) for m in safe})
                if self.preserve and parent in safe:
                    preserved.intersection_update(safe)
            if votes:
                usable += 1
                totals.update({m:n/len(trial) for m,n in votes.items()})
        return max((m for m in options if m in preserved),key=lambda m:(totals[m],m==parent)) if usable>=2 else parent


class PreservedExchanges(ExchangedRoads):
    preserve = True
