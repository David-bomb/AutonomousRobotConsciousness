"""Наведение на финиш, когда ни один заезд не может продлить оставшееся движение."""
from functools import lru_cache

import numpy as np

from .bridge_automaton import transitions
from .grid import destination
from .interface import Move
from .pattern_roads import PatternRoads
from .street_trunk import alternatives


def goal_deadline(view):
    f,r=view.fuel_distance,view.repair_distance
    if view.fuel<=view.health and (f is None or f>view.fuel):
        return True
    if view.health<=view.fuel and (r is None or r>view.health):
        return True
    can_fuel=f is not None and f<=view.fuel and f<view.health
    can_repair=r is not None and r<=view.health and r<view.fuel
    return not (can_fuel or can_repair)


@lru_cache(maxsize=128)
def goal_votes(memory,position,goal,horizon):
    h,w=len(memory),len(memory[0])
    index=lambda p:p[1]*w+p[0]
    origin,end=index(position),index(goal)
    moves=tuple(Move)
    choices={m:(i,index(q)) for i,m in enumerate(moves) if (q:=destination(memory,position,m))}
    out=dict.fromkeys(choices,0.)
    incoming=np.arange(len(moves))[:,None]
    visible=(np.frombuffer("".join(memory).encode(),dtype="S1")!=b"?").reshape(h,w)
    neighborhood=np.lib.stride_tricks.sliding_window_view(np.pad(visible,1,constant_values=True),(3,3))
    observed=neighborhood.all(axis=(2,3)).ravel()
    for weight,grid in alternatives(memory,goal):
        neighbors,turns=transitions(grid,False)
        values=np.zeros((len(moves),h*w+1))
        values[:,end]=1.
        for _ in range(max(0,horizon-1)):
            forward=values[incoming,neighbors]
            expected=np.einsum("ijn,jn->in",turns,forward)
            # По уже известным дорогам робот может сам выбрать дальнейшую
            # ветвь. В скрытой области учитывается несколько продолжений.
            known=np.where(turns[:,:,observed]>0,forward[:,observed][None,:,:],0.).max(axis=1)
            expected[:,observed]=known
            values[:,:-1]=expected
            values[:,end]=1.
            values[:,origin]=0.
        for m,(j,q) in choices.items():
            out[m]+=weight*values[j,q]
    return out


class DeadlineGoal(PatternRoads):
    def choose(self,view):
        move=super().choose(view)
        if not goal_deadline(view) or self.activity=="Подтверждённый путь к финишу":
            return move
        x,y=view.position
        votes=goal_votes(view.memory,view.position,(x+view.dx,y+view.dy),min(view.fuel,view.health))
        if any(votes.values()):
            revised=max(votes,key=lambda m:(votes[m],m==move))
            if revised!=move:
                self.activity="Прямой путь в оставшиеся ходы"
                return revised
        return move
