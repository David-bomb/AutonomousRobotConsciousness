"""Сколько вариантов следующего окна совместимы с реальными измерениями."""
from functools import lru_cache
from itertools import product

from .cell_hypotheses import occupancy
from .exact_worlds import layers
from .grid import destination
from .interface import Move
from .route_witness import geometry
from .street_trunk import StreetTrunk


@lru_cache(maxsize=128)
def window_votes(memory,origin,goal,readings,prior,old,sites):
    h,w=len(memory),len(memory[0])
    ix=lambda p:p[1]*w+p[0]
    shifts,_=geometry(w,h)
    terrain="".join(memory)
    known=sum(1<<i for i,k in enumerate(terrain) if k not in "#?") | (1<<ix(goal))
    allowed=sum(1<<i for i,k in enumerate(terrain) if k!="#")
    readings_at=[(1<<ix(origin),readings)]
    if prior is not None:
        readings_at.append((1<<ix(prior),old))
    sources=[sum(1<<ix(p) for p in group) for group in sites]
    _,chance=occupancy(memory)
    neighbors={m:q for m in Move if (q:=destination(memory,origin,m)) is not None}
    votes=[dict.fromkeys(neighbors,0.) for _ in readings]
    for move,(x,y) in neighbors.items():
        positions=[j*w+i for j in range(max(0,y-1),min(h,y+2)) for i in range(max(0,x-1),min(w,x+2))
                   if terrain[j*w+i]=="?" and (i,j)!=goal]
        assert len(positions)<=5
        for proposed in product((False,True),repeat=len(positions)):
            lower,upper=known,allowed
            weight=1.
            for j,road in zip(positions,proposed):
                if road:
                    lower|=1<<j
                    weight*=chance[j]
                else:
                    upper&=~(1<<j)
                    weight*=1-chance[j]
            shares=[]
            for k,(target,d) in enumerate(zip(sources,readings)):
                if not d or not target:
                    shares.append(0.)
                    continue
                radius=max(v[k] or 0 for _,v in readings_at)
                pessimistic=layers(lower,shifts,target,radius)
                optimistic=layers(upper,shifts,target,radius)
                if any(v[k] is not None and (not (optimistic[v[k]] & pt) or
                       (v[k]>0 and pessimistic[v[k]-1] & pt)) for pt,v in readings_at):
                    break
                possible=[m for m,p in neighbors.items() if optimistic[d-1] & (1<<ix(p))]
                if not possible:
                    break
                certain=bool(pessimistic[d-1] & (1<<ix((x,y))))
                shares.append(1. if certain else float(move in possible)/len(possible))
            else:
                for row,share in zip(votes,shares):
                    row[move]+=weight*share
    return votes


class WindowBounds(StreetTrunk):
    def direction_votes(self,view,readings,old,prior,sites):
        baseline=super().direction_votes(view,readings,old,prior,sites)
        x,y=view.position
        forecast=window_votes(view.memory,view.position,(x+view.dx,y+view.dy),readings,prior,old,sites)
        result=[]
        for parent,extra in zip(baseline,forecast):
            row={m:s*extra.get(m,0.) for m,s in parent.items()}
            result.append(row if any(row.values()) else parent)
        return result
