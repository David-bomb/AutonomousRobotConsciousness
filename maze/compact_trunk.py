"""Проверка минимального управления до раскрытия возобновляемой пары станций."""
from .frontier_automaton import _paths
from .resource_automaton import confirmed_route,service_order
from .street_trunk import StreetTrunk


class CompactTrunk(StreetTrunk):
    def _before_base(self,view):
        x,y=view.position
        goal=x+view.dx,y+view.dy
        route=confirmed_route(view,goal,_paths(view.memory,view.position))
        old=self._fallback.last_lengths.copy()
        returning=self._fallback.returning
        service=service_order(view,self._fallback.mode,self.RESERVE)
        move=self._fallback.choose(view)
        self.activity=self._fallback.mode.value
        if route is not None:
            return move
        return self.refine_approach(view,service,old,returning,move)
