"""Ищет наблюдательно неразличимые первые решения с несовместимыми успешными ходами."""
from concurrent.futures import ProcessPoolExecutor
import json
from dataclasses import replace

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,transition
from maze.environment import Environment
from maze.grid import destination,locations
from scripts.audit_automaton import Reachability


def balls(opened,width,height,sites,radii):
    left=sum(1<<(y*width) for y in range(height))
    right=left<<(width-1)
    n=lambda v:v>>width
    s=lambda v:v<<width
    e=lambda v:(v & ~right)<<1
    w=lambda v:(v & ~left)>>1
    corners=[(a,b,a(opened)&b(opened)) for a in (n,s) for b in (e,w)]
    current=sites
    out={}
    for d in range(max(radii)+1):
        if d in radii:
            out[d]=current
        reach=n(current)|s(current)|e(current)|w(current)
        for a,b,legal in corners:
            reach|=a(b(current)) & legal
        current|=reach & opened
    return out


def inspect(job):
    name,level,loss=job
    env=Environment(level)
    v=env.view
    w,h=len(level.grid[0]),len(level.grid)
    bit=lambda p:1<<(p[1]*w+p[0])
    start=bit(v.position)
    current=(v.position,v.fuel,v.health)
    known={p for p in (level.start,level.goal)}
    hidden={(x,y) for y,row in enumerate(v.memory) for x,c in enumerate(row)
            if c=="?" and (x,y) not in known and 0<x<w-1 and 0<y<h-1}
    opened=sum(bit((x,y)) for y,row in enumerate(level.grid) for x,c in enumerate(row) if c!="#")
    markers={k:sum(bit(p) for p in locations(level.grid,k)) for k in "FR"}
    readings={"F":v.fuel_distance,"R":v.repair_distance}
    stocks={"F":v.fuel,"R":v.health}
    neighbors={m.name:destination(level.grid,v.position,m) for m in MOVES}
    old_viable=set(loss["viable"])
    old_mask=sum(bit(neighbors[m]) for m in old_viable)
    chosen=bit(neighbors[loss["move"]])
    attempts=0
    for kind in "FR":
        d=readings[kind]
        if d is None or stocks[kind]-d not in (0,1):
            continue
        field=env._lengths[kind]
        closing=hidden.copy()
        # При одном закрытии оно должно разорвать хотя бы этот конкретный
        # короткий путь из каждого ранее годного направления.
        for m in old_viable:
            p=neighbors[m]
            if field[p]>stocks[kind]-1:
                continue
            segment=set()
            while field[p]:
                action,q=next((action,vv) for action in MOVES
                               if (vv:=destination(level.grid,p,action)) and field[vv]==field[p]-1)
                segment.update((p,q))
                dx,dy=action.value
                if dx and dy:
                    segment.update(((p[0]+dx,p[1]),(p[0],p[1]+dy)))
                p=q
            closing &= segment
        closing={p for p in closing if level.grid[p[1]][p[0]]=="."}
        other="R" if kind=="F" else "F"
        radii={d,d-1,stocks[kind]-1}
        for a in sorted(hidden):
            if level.grid[a[1]][a[0]]!="#":
                continue
            added=opened|bit(a)
            if not balls(added,w,h,markers[kind],{stocks[kind]-1})[stocks[kind]-1]&chosen:
                continue
            for b in sorted(closing):
                attempts+=1
                mask=added & ~bit(b)
                dist=balls(mask,w,h,markers[kind],radii)
                near=dist[stocks[kind]-1]
                if not near&chosen or near&old_mask or not dist[d]&start or dist[d-1]&start:
                    continue
                od=readings[other]
                if od is not None:
                    sec=balls(mask,w,h,markers[other],{od,od-1})
                    if not sec[od]&start or sec[od-1]&start:
                        continue
                grid=[list(row) for row in level.grid]
                grid[a[1]][a[0]]="."
                grid[b[1]][b[0]]="#"
                alt=replace(level,grid=tuple("".join(row) for row in grid))
                assert Environment(alt).view==v
                oracle=PhysicalRoutes(alt)
                viable={m.name for m in MOVES if transition(oracle,current,m,env.limit-1)}
                if viable & old_viable or loss["move"] not in viable:
                    continue
                for same,expected in ((level,old_viable),(alt,viable)):
                    exact=Reachability(same)
                    assert expected=={m.name for m,q in exact.edges[v.position]
                                      if exact.possible(q,*exact.after(q,v.fuel,v.health),env.limit-1)}
                return dict(map=name,checked=attempts,open=a,close=b,actual=sorted(old_viable),
                            alternative=sorted(viable),move=loss["move"],fuel=v.fuel,health=v.health,
                            dF=v.fuel_distance,dR=v.repair_distance)
    return dict(map=name,checked=attempts,found=False)


def main():
    rows=[json.loads(s) for s in (HERE/"range--range-roads.jsonl").read_bytes().splitlines()]
    lookup={c[1]:c[-1] for c in cases()}
    jobs=[(r["map"],lookup[r["map"]],r["child_loss"]) for r in rows
          if r.get("child_loss",{}).get("step")==1 and r["split"]!="val"]
    with (HERE/"twins.jsonl").open("x",encoding="utf8") as out,ProcessPoolExecutor(2) as pool:
        for r in pool.map(inspect,jobs):
            out.write(json.dumps(r)+"\n")
            out.flush()
    print(len(jobs),"первых необратимых решений")


if __name__=="__main__":
    main()
