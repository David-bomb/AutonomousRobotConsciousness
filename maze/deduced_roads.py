"""Вывод обязательных дорог и стен из точных показаний без обращения к среде."""
from dataclasses import replace
from functools import lru_cache

from .exact_worlds import edges,layers
from .route_witness import geometry
from .street_trunk import StreetTrunk


def bits(mask):
    while mask:
        bit=mask & -mask
        yield bit
        mask^=bit


@lru_cache(maxsize=128)
def deductions(memory,position,goal,readings,prior,old,sites,check_consistency=False):
    h,w=len(memory),len(memory[0])
    index=lambda p:p[1]*w+p[0]
    known=sum(1<<i for i,k in enumerate("".join(memory)) if k not in "#?")|(1<<index(goal))
    upper=sum(1<<i for i,k in enumerate("".join(memory)) if k!="#")
    initial_low,initial_high=known,upper
    shifts,_=geometry(w,h)
    links=edges(w,h)
    conditions=[]
    for sites_of_kind,d,previous in zip(sites,readings,old):
        requirements=[]
        if check_consistency and d is not None and ((d==0)!=(position in sites_of_kind)):
            return None
        if d:
            requirements.append((index(position),d))
        if prior is not None and previous:
            requirements.append((index(prior),previous))
        if requirements:
            source=sum(1<<index(p) for p in sites_of_kind)
            conditions.append((source,requirements,max(t for _,t in requirements)))
    if not conditions:
        return memory

    def too_short(opened):
        for source,requirements,limit in conditions:
            field=layers(opened,shifts,source,limit-1)
            if any(field[d-1] & (1<<p) for p,d in requirements):
                return True
        return False

    def too_long(opened):
        for source,requirements,limit in conditions:
            field=layers(opened,shifts,source,limit)
            if any(not field[d] & (1<<p) for p,d in requirements):
                return True
        return False

    if check_consistency and (too_short(known) or too_long(upper)):
        return None
    for _ in range(3):
        unknown=upper & ~known
        # Первая неизвестная клетка, открывающая короткий путь, должна
        # соединить хотя бы две уже открытые стороны.
        n,e,s,w_side=(shift(known) for shift in shifts)
        neighbors=(n & (e|s|w_side)) | (e & (s|w_side)) | (s & w_side)
        walls=sum(bit for bit in bits(unknown & neighbors) if too_short(known|bit))
        upper &= ~walls
        # Одного оптимистичного пути на каждое показание достаточно для
        # отбора кандидатов. Закрытие клетки вне этого пути его сохраняет.
        witness=0
        for source,requirements,limit in conditions:
            field=layers(upper,shifts,source,limit)
            for p,d in requirements:
                if not field[d] & (1<<p):
                    continue
                length=next(t for t,ball in enumerate(field) if ball & (1<<p))
                while length:
                    p,gate=next((q,gate) for q,gate in links[p]
                                if gate & upper==gate and field[length-1] & (1<<q))
                    witness |= gate
                    length-=1
        roads=sum(bit for bit in bits(witness & upper & ~known) if too_long(upper & ~bit))
        known |= roads
        if check_consistency and (too_short(known) or too_long(upper)):
            return None
        if not (roads or walls):
            break
    added,closed=known & ~initial_low,initial_high & ~upper
    if not (added or closed):
        return memory
    board=list("".join(memory))
    for mask,kind in ((added,"."),(closed,"#")):
        for bit in bits(mask):
            board[bit.bit_length()-1]=kind
    return tuple("".join(board[j*w:(j+1)*w]) for j in range(h))


class DeducedRoads(StreetTrunk):
    def direction_votes(self,view,readings,old,prior,sites):
        x,y=view.position
        grid=deductions(view.memory,view.position,(x+view.dx,y+view.dy),readings,prior,old,sites)
        return super().direction_votes(replace(view,memory=grid),readings,old,prior,sites)
