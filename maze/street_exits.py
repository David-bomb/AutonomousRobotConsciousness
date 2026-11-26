"""При выборе окна к финишу учитывать сплошные стены между замеченными рядами дорог."""
from functools import lru_cache

import numpy as np

from .pattern_roads import patterns
from .perimeter_paths import interior_graph, wall_probability
from .simple_exit_paths import simple_exit_probabilities
from .standby_turn import RankedStandby


@lru_cache(maxsize=32)
def structured_curves(memory, goal, points, boundary):
    original = simple_exit_probabilities(memory,goal,points)
    board,models,weights = patterns(memory,goal)
    h,w = len(memory),len(memory[0])
    y,x = np.indices((h,w)).reshape(2,-1)
    road = (board!=b"#") & (board!=b"?")
    road[goal[1]*w+goal[0]] = True
    result = {p:c.copy() for p,c in original.items()}
    for j in np.argsort(-weights)[:4]:
        if models[j] is None:
            continue
        sx,sy,ox,oy = models[j]
        between = (x%sx!=ox) & (y%sy!=oy)
        if (between & road).any():
            continue
        n = (between & (board==b"#")).sum()
        if not n:
            continue
        # Гипотезу о стенах применяем лишь при отсутствии известного контрпримера.
        # При равном начальном доверии сплошная стена выигрывает у неизвестной
        # доли стен в n+1 раз после наблюдения n стен.
        weight = weights[j]*(n+1)/(n+2)
        guess = board.copy()
        guess[between] = b"#"
        assumed = tuple(row.tobytes().decode() for row in guess.reshape(h,w))
        curves = simple_exit_probabilities(assumed,goal,points)
        if not any(c[-1] for c in curves.values()):
            continue
        perimeter = wall_probability(memory,goal) if boundary else 0.
        if perimeter:
            enclosed = simple_exit_probabilities(assumed,goal,points,interior_graph)
            if any(c[-1] for c in enclosed.values()):
                curves = {p:(1-perimeter)*c + perimeter*enclosed[p] for p,c in curves.items()}
        for p in points:
            result[p] += weight*(curves[p]-original[p])
    return result


class StreetExits(RankedStandby):
    use_boundary = False

    def sortie_curves(self,view,goal,points):
        return structured_curves(view.memory,goal,points,self.use_boundary)


class BoundedStreetExits(StreetExits):
    use_boundary = True
