"""Строить варианты неизвестных клеток, объясняющие оба датчика без погрешности."""
from functools import lru_cache

import numpy as np

from .cell_hypotheses import occupancy
from .grid import destination
from .interface import Move
from .pattern_roads import MarkedStreets
from .route_witness import geometry


@lru_cache(maxsize=32)
def edges(width,height):
    out=[]
    for y in range(height):
        for x in range(width):
            row=[]
            for m in Move:
                dx,dy=m.value
                a,b=x+dx,y+dy
                if not (0<=a<width and 0<=b<height):
                    continue
                j=b*width+a
                required=1<<j
                if dx and dy:
                    required|=(1<<(y*width+a))|(1<<(b*width+x))
                row.append((j,required))
            out.append(row)
    return out


def layers(opened,shifts,source,limit):
    n,e,s,w=shifts
    corners=[(a,b,a(opened)&b(opened)) for a in (n,s) for b in (e,w)]
    out=[source]
    for _ in range(limit):
        last=out[-1]
        step=n(last)|e(last)|s(last)|w(last)
        for a,b,mask in corners:
            step|=a(b(last)) & mask
        new=last|(step & opened)
        out.append(new)
        if new==last:
            out.extend([last]*(limit+1-len(out)))
            break
    return out


def pack(mask):
    return int.from_bytes(np.packbits(mask,bitorder="little").tobytes(),"little")


def opening_on_path(field,p,d,opened,mutable,links,rng):
    candidates=0
    while d:
        choices=[(q,gate) for q,gate in links[p] if gate & opened==gate and field[d-1] & (1<<q)]
        p,gate=choices[int(rng.integers(len(choices)))]
        candidates|=gate
        d-=1
    return candidates & mutable


@lru_cache(maxsize=96)
def exact_votes(view,readings,old,prior,sites):
    if any(d is None or not 0<d<=s for d,s in zip(readings,(view.fuel,view.health))):
        return None
    grid=view.memory
    h,w=len(grid),len(grid[0])
    idx=lambda p:p[1]*w+p[0]
    start=idx(view.position)
    req=[[(start,d)] for d in readings]
    for i,prev in enumerate(old):
        if prior is not None and prev is not None:
            req[i].append((idx(prior),prev))
    radii=[max(d for _,d in r) for r in req]
    marks=[sum(1<<idx(p) for p in group) for group in sites]
    board,prob=occupancy(grid)
    fixed=(board!=b"?") & (board!=b"#")
    x,y=view.position
    fixed[(y+view.dy)*w+x+view.dx]=True
    mutable=pack((board==b"?") & ~fixed)
    shifts,_=geometry(w,h)
    links=edges(w,h)
    choices={m:1<<idx(q) for m in Move if (q:=destination(grid,view.position,m))}
    votes=[dict.fromkeys(choices,0.) for _ in range(2)]
    def assess(mask):
        fields=[]
        short=[]
        for i,d in enumerate(radii):
            field=layers(mask,shifts,marks[i],d)
            fields.append(field)
            for p,length in req[i]:
                b=1<<p
                if not field[length] & b:
                    return None
                if length and field[length-1] & b:
                    true=next(j for j,ball in enumerate(field) if ball & b)
                    short.append((length-true,i,p,true))
        return fields,short
    rng=np.random.default_rng(42)
    matched=0
    for _ in range(16):
        u=rng.random(len(board))
        rank=(u/(1-u))*((1-prob)/prob)
        values=np.unique(np.r_[0.,rank[board==b"?"]])
        lo,hi=0,len(values)-1
        while lo<hi:
            mid=(lo+hi)//2
            sample=fixed|((board==b"?")&(rank<=values[mid]))
            if assess(pack(sample)) is None:
                lo=mid+1
            else:
                hi=mid
        opened=pack(fixed|((board==b"?")&(rank<=values[lo])))
        actual=assess(opened)
        if actual is None:
            continue
        removable=opened & mutable
        for _ in range(2*max(radii)+1):
            fields,short=actual
            if not short:
                matched+=1
                for i,d in enumerate(readings):
                    descending=[m for m,bit in choices.items() if fields[i][d-1] & bit]
                    for m in descending:
                        votes[i][m]+=1/len(descending)
                break
            _,i,p,d=max(short)
            options=opening_on_path(fields[i],p,d,opened,removable,links,rng)
            bits=[]
            while options:
                bit=options & -options
                bits.append(bit)
                options^=bit
            for j in rng.permutation(len(bits)):
                bit=bits[j]
                removable &= ~bit
                trial=assess(opened & ~bit)
                if trial is not None:
                    opened &= ~bit
                    actual=trial
                    break
            else:
                break
        if matched>=6:
            break
    return votes if matched else None


class ExactWorlds(MarkedStreets):
    def direction_votes(self,view,readings,old,prior,sites):
        return exact_votes(view,readings,old,prior,sites) or super().direction_votes(view,readings,old,prior,sites)
