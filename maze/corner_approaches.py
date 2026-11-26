"""Учитывать стены, без которых предполагаемый путь к станции оказался бы короче замера."""
from .frontier_reserve import _lengths, _optimistic
from .path_particles import particle_choice
from .standby_turn import RankedStandby


class CornerApproaches(RankedStandby):
    multiple = False

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        measured = readings[kind]
        if measured is not None and 0<=(view.fuel,view.health)[kind]-measured<2:
            lower = _lengths(_optimistic(view.memory),view.position)
            plausible = [p for p in sites[kind] if lower.get(p,float("inf"))<=measured]
            if len(plausible)==1 or self.multiple and plausible:
                raw = self.direction_votes(view,readings,old,prior,sites)[kind]
                return particle_choice(view,readings,old,prior,sites,options,kind,default,raw,corner_walls=True)
        return super().rank_directions(view,readings,old,prior,sites,options,kind,default)


class MultipleCornerApproaches(CornerApproaches):
    multiple = True
