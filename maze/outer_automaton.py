"""После последнего выезда дорога к цели проходит за границей известной области."""
from collections import deque
from functools import lru_cache, partial

from .frontier_automaton import _paths
from .frontier_reserve import _lengths, _optimistic
from .grid import destination
from .interface import Move
from .road_hypotheses import RangeRoads


@lru_cache(maxsize=128)
def outward(memory,goal,stops,root,allow_service=False):
    component = _paths(memory,root)
    if goal in component:
        return None
    grid = _optimistic(memory)
    to_goal = _lengths(grid,goal)
    targets = [(goal,"B")]
    if allow_service:
        targets.extend((p,k) for p,k in stops if p not in component and p in to_goal)
    estimates = {}
    for target,kind in targets:
        queue = deque([(target,0)])
        seen = {target}
        lengths = {}
        while queue:
            p,d = queue.popleft()
            for m in Move:
                q = destination(grid,p,m)
                if q is None or q in seen:
                    continue
                if q in component:
                    # Волна пришла к последнему месту выезда с внешней стороны.
                    lengths.setdefault(q,d+1)
                else:
                    seen.add(q)
                    queue.append((q,d+1))
        for point,leg in lengths.items():
            if kind=="B":
                plan = leg,leg,100
            elif kind=="F":
                remaining = to_goal[target]
                plan = leg,leg+remaining,100-remaining
            else:
                remaining = to_goal[target]
                plan = leg+remaining,leg,100-remaining
            estimates.setdefault(point,[]).append(plan)
    return estimates or None


class OuterRange(RangeRoads):
    outer_rank = staticmethod(outward)


class OuterService(OuterRange):
    outer_rank = staticmethod(partial(outward,allow_service=True))
