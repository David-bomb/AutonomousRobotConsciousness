"""Использовать физически подтверждённый спуск ко второй станции для выбора первого хода."""
from .grid import destination
from .interface import Move
from .probe_pairs import paired_directions
from .proven_approach import service_field
from .resource_automaton import opposite
from .unseen_approach import UnseenReturn


def favor_given(matrix, options, anchors, parent):
    moves = tuple(Move)
    columns = [moves.index(m) for m in anchors]
    marginal = matrix.sum(axis=1)
    conditional = matrix[:,columns].sum(axis=1)
    winner = max(options,key=lambda m:(conditional[moves.index(m)],m==parent))
    best,old = moves.index(winner),moves.index(parent)
    if conditional[best]*marginal[old] > conditional[old]*marginal[best]:
        return winner
    return parent


class AnchoredDirections(UnseenReturn):
    known_routes = False

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        other = 1-kind
        stocks = view.fuel,view.health
        if (None in readings or not readings[other] or len(options)<2 or
                not 0 <= min(stocks[kind],stocks[other]-1)-readings[kind] < 2):
            return parent
        certain = set()
        if prior is not None and old[other] == readings[other]-1:
            certain.add(opposite(view.last_move))
        if self.known_routes:
            distances = service_field(view.memory,sites[other])
            for m in Move:
                q = destination(view.memory,view.position,m)
                if q is not None and distances.get(q) == readings[other]-1:
                    certain.add(m)
        if not certain:
            return parent
        pairs = paired_directions(view,readings,old,prior,sites)
        return favor_given(pairs.T if kind else pairs,options,certain,parent)


class MappedDirections(AnchoredDirections):
    known_routes = True
