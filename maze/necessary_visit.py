"""Выявить по геометрии памяти клетки, без посещения которых её нельзя было заполнить."""
from functools import lru_cache

from .grid import destination
from .interface import Move
from .sight_automaton import window_unknown
from .unseen_approach import UnseenReturn


@lru_cache(maxsize=96)
def possible_centers(memory):
    return frozenset((x,y) for y,row in enumerate(memory) for x,c in enumerate(row)
                     if c not in "#?" and not window_unknown(memory,(x,y)))


@lru_cache(maxsize=256)
def necessary_visit(memory, origin, excluded):
    centers = possible_centers(memory)
    if excluded not in centers or origin not in centers:
        return False
    if origin == excluded:
        return True
    reached = {origin}
    pending = [origin]
    while pending:
        p = pending.pop()
        for m in Move:
            q = destination(memory, p, m)
            if q != excluded and q in centers and q not in reached:
                reached.add(q)
                pending.append(q)
    # Все наблюдавшиеся стены и обычные дороги должны попадать хотя бы в
    # одно окно возможного пути. Сервисы и координаты финиша известны отдельно.
    for y,row in enumerate(memory):
        for x,c in enumerate(row):
            if c in "#." and not any((x+i,y+j) in reached for i in (-1,0,1) for j in (-1,0,1)):
                return True
    return False


class NecessaryReturn(UnseenReturn):
    approach_label = "Подход с учётом обследованной области"

    def alternatives(self, view, options):
        return {m:d for m,d in options.items() if not necessary_visit(
            view.memory, view.position, destination(view.memory, view.position, m))}


class NecessaryApproach(NecessaryReturn):
    after_return_only = False
