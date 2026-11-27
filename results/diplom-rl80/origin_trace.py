"""Bound possible starts and elapsed travel from the native cell-memory footprint."""
from collections import deque
from functools import lru_cache
from math import inf


@lru_cache(maxsize=512)
def footprint(memory,position,prior_at,first):
    width=len(memory[0]);height=len(memory)
    def index(x,y):return y*width+x
    exposed={index(x,y) for y,row in enumerate(memory) for x,c in enumerate(row) if c not in "?FR"}
    # F/R labels are visible before travel. They do not certify a past visit.
    centers={};witness={}
    for y in range(1,height-1):
        for x in range(1,width-1):
            if memory[y][x] in "?#":continue
            block=tuple(index(i,j) for j in range(y-1,y+2) for i in range(x-1,x+2))
            if any(memory[u//width][u%width]=="?" for u in block):continue
            p=index(x,y);centers[p]=(x,y)
            for u in block:witness.setdefault(u,[]).append(p)
    cur=index(*position)
    if first:return ((cur,0,0),)
    graph={u:[] for u in centers}
    for u,(x,y) in centers.items():
        for dx,dy in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            v=index(x+dx,y+dy)
            if v in centers and memory[y+dy][x]!="#" and memory[y][x+dx]!="#":graph[u].append(v)
    def distances(p):
        q=deque([p]);d={p:0}
        while q:
            u=q.popleft()
            for v in graph[u]:
                if v not in d:d[v]=d[u]+1;q.append(v)
        return d
    assert cur in centers
    prev=index(*prior_at)
    assert prev in graph
    # Every observed nonlandmark pixel was within a window. The current window
    # is already covered at the final step, so only others constrain its prefix.
    visited=distances(prev)
    before=exposed-{index(x,y) for y in range(position[1]-1,position[1]+2)
                               for x in range(position[0]-1,position[0]+2)}
    groups={tuple(u for u in witness[p] if u in visited) for p in before}
    assert all(groups)
    groups=[g for g in groups if not any(len(t)<len(g) and set(t).issubset(g) for t in groups)]
    mandatory={g[0] for g in groups if len(g)==1}|{prev}
    # Count bridges that a window-compatible walk must cross in each direction.
    tin={};low={};cut=[];clock=0
    def dfs(u,par):
        nonlocal clock
        tin[u]=low[u]=clock;clock+=1
        needed=u in mandatory
        for v in graph[u]:
            if v==par:continue
            if v in tin:low[u]=min(low[u],tin[v]);continue
            sub=dfs(v,u);needed|=sub
            low[u]=min(low[u],low[v])
            if low[v]>tin[u]:cut.append((tin[v],clock,sub))
        return needed
    dfs(prev,-1)
    result=[]
    for u in visited:
        x,y=centers[u]
        if memory[y][x] in "FRB":continue
        ds=distances(u)
        bridge=sum(1 if lo<=tin[u]<hi else 2*needed for lo,hi,needed in cut)
        through=max((min(ds[c]+visited[c] for c in g) for g in groups),default=0)
        lower=1+max(bridge,through,visited[u])
        result.append((u,lower,visited[u]+1))
    return tuple(result)


def origins(v):
    x,y=v.position
    if v.last_move is not None and v.moved:
        dx,dy=v.last_move.value;old=(x-dx,y-dy)
    else:old=v.position
    return footprint(v.memory,v.position,old,v.last_move is None)
