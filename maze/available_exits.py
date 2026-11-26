"""Сравнивать первые физические ходы по окнам, которые останутся достижимыми после хода."""
import numpy as np

from .first_window import first_events
from .frontier_automaton import _arrive, _frontiers
from .grid import cell, destination
from .serviced_windows import backup_stocks
from .standby_turn import RankedStandby


def remaining_shares(view,points,curves,first):
    q = destination(view.memory,view.position,first)
    after = _arrive(cell(view.memory,*q),1,view.fuel,view.health)
    if after is None:
        return np.zeros(len(points))
    options = backup_stocks(view.memory,q,*after,points,include_origin=True)
    # Новое наблюдение по дороге не мешает продолжить движение по уже известным клеткам.
    return np.array([max((curves[p][min(f,h)] for f,h in options.get(p,())),default=0.)
                     for p in points])


class AvailableExits(RankedStandby):
    keep_each = True

    def pick(self,view,goal,arrivals):
        choices = list(arrivals)
        parent = super().pick(view,goal,choices)
        actions = tuple(dict.fromkeys(a.first for a in choices))
        if len(actions)<2 or any(a.point==goal for a in choices):
            return parent
        points = tuple(sorted(p for p,_,_ in first_events(view.memory,view.position,frozenset(_frontiers(view.memory)))))
        curves = self.sortie_curves(view,goal,points)
        original = remaining_shares(view,points,curves,parent.first)
        best,score = parent.first,original.sum()
        for m in actions:
            if m==parent.first:
                continue
            shares = remaining_shares(view,points,curves,m)
            total = shares.sum()
            if total>score+1.e-12 and (not self.keep_each or (shares>=original-1.e-12).all()):
                best,score = m,total
        if best==parent.first:
            return parent
        return super().pick(view,goal,[a for a in choices if a.first==best])


class FlexibleExits(AvailableExits):
    keep_each = False
