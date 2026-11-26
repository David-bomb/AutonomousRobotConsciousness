"""После закрытого окна выполнить один прямой выезд, оплаченный расчётом запасного пути."""
from .frontier_automaton import _frontiers
from .grid import destination
from .interface import Move
from .second_window import SecondWindow
from .specific_paths import SpecificPaths


def degree(memory, point):
    return sum(destination(memory, point, m) is not None for m in Move)


class ProbeThenDirect(SecondWindow):
    def __init__(self):
        super().__init__()
        self.expected_degree = None
        self.direct_once = False

    def choose(self, view):
        if view.last_move is None:
            self.direct_once = False
        elif self.expected_degree is not None and view.moved:
            # После осмотра доступны все ходы из фактической клетки. Одного
            # числа от 0 до 8 достаточно, чтобы узнать, открылся ли новый ход.
            self.direct_once = not self.direct_once and degree(view.memory, view.position) == self.expected_degree
        self.expected_degree = None
        self.choosing_window = False
        move = super().choose(view)
        if self.choosing_window:
            q = destination(view.memory, view.position, move)
            if q in _frontiers(view.memory):
                self.expected_degree = degree(view.memory, q)
        else:
            self.direct_once = False
        return move

    def pick(self, view, goal, arrivals):
        self.choosing_window = True
        if self.direct_once:
            return SpecificPaths.pick(self, view, goal, arrivals)
        return super().pick(view, goal, arrivals)
