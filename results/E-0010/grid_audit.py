"""Сверка волнового поиска с прежним обходом и измерение процессорного времени."""
from collections import deque
import json
from statistics import median
from time import process_time

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment
from maze.grid import _adjacency, destination, locations, wave


def original(grid,starts,stop_at):
    seen=set(starts)
    pending=deque((p,0,None) for p in sorted(seen))
    while pending:
        p,d,first=pending.popleft()
        yield p,d,first
        if d and p in stop_at:
            continue
        for m in MOVES:
            q=destination(grid,p,m)
            if q is not None and q not in seen:
                seen.add(q)
                pending.append((q,d+1,first or m))


def main():
    runs=read("street-trunk")
    samples=[]
    for split,name,_,_,level in cases():
        if name not in {"H1-05","fresh-H1-11","check-H4-12","val-3-10"}:
            continue
        env=Environment(level)
        grids=[env.view.memory]
        for char in runs[name]["actions"][:12]:
            env.step(MOVES[int(char)])
        grids.extend((env.view.memory,level.grid))
        for grid in grids:
            stops=frozenset(p for k in "FR" for p in locations(grid,k))
            origins=[(level.start,),(env.position,),tuple(sorted(stops)),(),((-1,0),)]
            for sources in origins:
                for ends in (frozenset(),stops):
                    samples.append((grid,sources,ends))
    assert len(samples)==120
    expected=[list(original(grid,sources,ends)) for grid,sources,ends in samples]
    times={"original":[],"cached":[]}
    for order in (("original","cached"),("cached","original"),("original","cached")):
        for version in order:
            _adjacency.cache_clear()
            start=process_time()
            for i,(grid,sources,ends) in enumerate(samples):
                walk=original(grid,sources,ends) if version=="original" else wave(grid,sources,stop_at=ends)
                if list(walk)!=expected[i]:
                    raise ValueError(f"Расхождение обхода {i}")
            times[version].append(process_time()-start)
    result={"searches":len(samples),"cells_checked":sum(map(len,expected)),
            "cpu_seconds":times,"median_cpu_seconds":{k:median(v) for k,v in times.items()}}
    (HERE/"grid-audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result,flush=True)


if __name__=="__main__":
    main()
