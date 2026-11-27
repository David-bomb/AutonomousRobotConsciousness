"""Гипотеза о комнатах по одному снимку памяти, известным сервисам и положению B."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..model import LIB

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=128)
def layout(w,h):
    centers=np.array([(x,y) for y in range(3,h-3,4) for x in range(3,w-3,4)],dtype=int)
    def coords(offsets):
        return np.stack([centers+off for off in offsets],1)
    core=coords([(x,y) for y in (-1,0,1) for x in (-1,0,1)])
    corners=coords([(x,y) for y in (-1,1) for x in (-1,1)])
    sides=coords([(x,y) for y in (-1,0,1) for x in (-2,2)])
    yy,xx=np.indices((h,w))
    inner=(xx>0)&(yy>0)&(xx<w-1)&(yy<h-1)
    even=inner&(xx%2==0)&(yy%2==0)
    return core,corners,sides,inner,even,xx,yy


def at(array,xy):
    return array[xy[...,1],xy[...,0]]


def representation(board,meta,w,h,count_table):
    core,corners,sides,inner,even,xx,yy=layout(w,h)
    floor=board[0,:h,:w]
    evidence=floor.copy()
    # Координаты публикуются штатным Observation даже до открытия соседних клеток.
    for point in meta[1:6]:
        y,x=divmod(int(point),40)
        evidence[y,x]=1
    possible=(at(evidence,core)!=2).all(1)
    witness=(at(evidence,corners)==1).any(1)
    wide=(at(evidence,sides)!=2).all(1)
    expected,upper=count_table[h,w]
    possible_cells=np.zeros_like(floor,bool)
    possible_cells[corners[...,1],corners[...,0]]=True
    outside=int(((evidence==1)&even&~possible_cells).sum())
    conflict=int((witness&~possible).sum())+outside
    filled=floor.copy();core_added=np.zeros_like(floor,bool)
    if not conflict and witness.sum()<=upper and possible.sum()>=expected-.51:
        allowed=possible & (witness if witness.sum()>=upper else True)
        remaining=np.zeros_like(floor,bool)
        xy=corners[allowed]
        remaining[xy[...,1],xy[...,0]]=True
        filled[(floor==0)&even&~remaining]=2
        xy=core[witness]
        core_added[xy[...,1],xy[...,0]]=True
        core_added &= floor==0
        filled[core_added]=1
    walls=(filled==2)&(floor==0)
    stats=[w/40,h/40,expected/12,upper/12,len(possible)/64,possible.mean(),witness.sum()/12,
           (witness&wide).sum()/12,min(conflict,5)/5,outside/5,(expected-witness.sum())/12,
           (possible.sum()-expected)/48,core_added.sum()/40,walls.sum()/300,
           (evidence[inner]!=0).mean(),((board[1,:h,:w]!=0)&inner).sum()/400]
    for a,b in ((0,0),(0,1),(1,0),(1,1)):
        region=inner&(xx%2==a)&(yy%2==b)
        opened=evidence[region]
        stats.extend(((opened==1).sum()/max(1,(opened!=0).sum()),(opened!=0).mean()))
    p=meta[0]
    for k in (0,1):
        points=meta[1+k*2:3+k*2]
        geometric=np.min(abs(points%40-p%40)+abs(points//40-p//40))
        stats.extend((meta[6+k]/45,(meta[6+k]-geometric)/30))
    stats.extend((sum((point%40)%2==0 and (point//40)%2==0 for point in meta[1:5])/4,
                  int((meta[5]%40)%2==0 and (meta[5]//40)%2==0),witness.any(),conflict==0))
    stats=np.array(stats,np.float32)
    assert stats.shape==(32,)
    if not (core_added.any() or walls.any()):
        return board,stats,core_added,walls
    grid=np.array([b"?",b".",b"#"],dtype="S1")[filled]
    result=np.empty_like(board)
    pos=int(p//40)*w+int(p%40)
    LIB.region(grid.tobytes(),w,h,pos,result)
    return result,stats,core_added,walls
