"""Построение и допуск сценариев E-0012 через штатный Environment."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time
import zlib

from maze.datasets import DATA, development_sets, geometry
from maze.environment import Environment, Status
from maze.grid import destination
from maze.interface import Move
from maze.levels import Level
from .geometry import FAMILIES, SIZES, generate
from .session import OUT, check_time, log

MOVES = tuple(Move)
SPLITS = ("train","validation_id","shift_development","test_id","test_ood")
TARGETS = dict(zip(SPLITS,(60000,3000,1500,6000,6000)))
DB = OUT/"corpus.sqlite"


def connect():
    db = sqlite3.connect(DB, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS geoms (split TEXT, idx INTEGER, hash TEXT UNIQUE, "
               "family TEXT, size INTEGER, seed INTEGER, group_name TEXT, data BLOB, "
               "PRIMARY KEY(split,idx))")
    return db


def design(split, i):
    family, size = FAMILIES[i%6], SIZES[(i//6)%5]
    group = "standard"
    if split == "shift_development":
        group = ("size","parameters","services","resources","combined")[i%5]
        family, size = FAMILIES[(i//5)%6], SIZES[(i//30)%5]
        if group in ("size","combined"):
            size = (45,49)[(i//30)%2]
    elif split == "test_ood":
        group = ("size","generator","topology","services","resources","combined")[i%6]
        family, size = FAMILIES[(i//6)%6], SIZES[(i//36)%5]
        if group in ("size","combined"):
            size = (61,81)[(i//36)%2]
        if group in ("generator","combined"):
            family = ("wilson","division")[(i//6)%2]
        if group == "topology":
            family, size = ("dfs","braid")[(i//6)%2], (41,49)[(i//36)%2]
    return family,size,group


def edges(grid):
    return {(x,y): [q for m in MOVES if (q:=destination(grid,(x,y),m)) is not None]
            for y,row in enumerate(grid) for x,c in enumerate(row) if c == "."}


def tree(graph,start):
    parents,lengths = {start:None}, {start:0}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in graph[p]:
            if q not in parents:
                parents[q] = p
                lengths[q] = lengths[p]+1
                queue.append(q)
    return parents,lengths


def path(parents,end):
    result = []
    while end is not None:
        result.append(end)
        end = parents[end]
    return result[::-1]


def route_and_services(graph, rng, mode, group):
    start = rng.choice(list(graph))
    parents,lengths = tree(graph,start)
    maximum = max(lengths.values())
    if maximum < 7:
        raise ValueError("Малая связная область")
    lower = max(7,round(maximum*rng.uniform(.27,.60)))
    candidates = [p for p,d in lengths.items() if lower<=d<=max(lower+2,round(maximum*.95))]
    goal = rng.choice(candidates)
    direct = lengths[goal]
    route = path(parents,goal)
    if mode and rng.random() < .72:
        # Заезд может лежать в стороне от кратчайшей дороги к финишу.
        entry = rng.choice([p for p,d in lengths.items() if 2<=d<=max(3,direct//2)])
        back,_ = tree(graph,entry)
        route = path(parents,entry)[:-1] + path(back,goal)
    facilities = {}
    stocks = []
    for symbol,required in (("F",mode in (1,3)),("R",mode in (2,3))):
        maximum_first = min(75, direct-1 if required else len(route)-2)
        fraction = rng.uniform(.16,.42) if symbol=="F" else rng.uniform(.35,.65)
        ix = max(1,min(maximum_first,int(fraction*direct)))
        choices = [j for j in range(1,min(direct if required else len(route)-1, 95))
                   if route[j] not in facilities and route[j] not in (start,goal)]
        if not choices:
            raise ValueError("Нет клетки для обслуживания")
        first = min(choices,key=lambda j:abs(j-ix))
        facilities[route[first]] = symbol
        slack = rng.randint(0,2 if group in ("resources","combined") else max(3,direct//5))
        stock = min(direct-1,first+slack) if required else min(100,len(route)+rng.randint(3,30))
        stocks.append(stock)
        last = first
        while len(route)-1-last > 95:
            options = [j for j in range(last+30,min(last+92,len(route)-1))
                       if route[j] not in (start,goal) and facilities.get(route[j],symbol)==symbol]
            if not options:
                raise ValueError("Нет клетки для цепочки обслуживания")
            last = rng.choice(options)
            facilities[route[last]] = symbol
    # Известные станции также располагаются вне доказательного маршрута.
    spare = [p for p in graph if p not in facilities and p not in (start,goal)]
    if spare and rng.random()<.55:
        reference = rng.choice(list(facilities))
        if group in ("services","combined"):
            spare.sort(key=lambda p:max(abs(p[0]-reference[0]),abs(p[1]-reference[1])))
            spare = spare[:max(1,len(spare)//20)]
        facilities[rng.choice(spare)] = rng.choice("FR")
    return route,facilities,stocks,direct


def certify(level, moves):
    env = Environment(level)
    for code in moves:
        if env.status != Status.RUNNING:
            return None
        env.step(MOVES[int(code)])
        if not env.moved:
            return None
    if env.status != Status.SUCCESS:
        return None
    return dict(steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                final_fuel=env.fuel,final_health=env.health,status=env.status.name)


def scenario(grid,graph,mode,rng,name,group):
    for _ in range(30):
        try:
            route,sites,stocks,shortest = route_and_services(graph,rng,mode,group)
        except ValueError:
            continue
        cells = [list(row) for row in grid]
        for (x,y),s in [*sites.items(),(route[0],"A"),(route[-1],"B")]:
            cells[y][x] = s
        source = "\n".join([name," ".join(map(str,stocks)),*map("".join,cells)])
        moves = "".join(str(MOVES.index(Move((b[0]-a[0],b[1]-a[1])))) for a,b in zip(route,route[1:]))
        proof = certify(Level.parse(source),moves)
        if proof:
            return dict(source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),
                        moves=moves,proof=proof,direct=shortest,mode=mode)
    raise ValueError("Не удалось подтвердить четыре сценария")


def make(args):
    split,i,attempt = args
    check_time()
    family,size,group = design(split,i)
    seed = (SPLITS.index(split)+1)*10**11+i*1000+attempt
    rng = random.Random(seed)
    grid = generate(family,size,rng,sparse=group=="topology")
    graph = edges(grid)
    try:
        rows = [scenario(grid,graph,j,rng,f"{split}-{i:05}-{j}",group) for j in range(4)]
    except (ValueError,IndexError) as error:
        return str(error)
    payload = zlib.compress(json.dumps(rows,separators=(",",":")).encode(),3)
    return split,i,geometry(grid),family,size,seed,group,payload


def selfcheck():
    direct = Level.parse("финиш\n1 1\n#####\n#AB##\n#####")
    assert certify(direct,"1")["status"] == "SUCCESS"
    assert certify(Level.parse("ресурсы\n1 2\n######\n#AFRB#\n######"),"111")["repairs"] == 1
    assert certify(Level.parse("невозможно\n1 1\n#####\n#A.B#\n#####"),"11") is None
    assert certify(Level.parse("стена\n5 5\n#####\n#A#B#\n#####"),"11") is None
    log("validator_selfcheck",positive=2,negative=2)


def build(db,workers,split,target,excluded):
    existing = {r[0] for r in db.execute("SELECT idx FROM geoms WHERE split=? AND idx<?",(split,target))}
    needed = [i for i in range(target) if i not in existing]
    start = time.perf_counter()
    for pos in range(0,len(needed),30):
        check_time()
        todo = needed[pos:pos+30]
        attempt = 0
        while todo:
            retry = []
            for i,result in zip(todo,workers.map(make,((split,i,attempt) for i in todo),chunksize=2)):
                if isinstance(result,str) or result[2] in excluded:
                    retry.append(i)
                else:
                    try:
                        db.execute("INSERT INTO geoms VALUES (?,?,?,?,?,?,?,?)",result)
                    except sqlite3.IntegrityError:
                        retry.append(i)
            if retry:
                log("geometry_retry",split=split,indices=retry,attempt=attempt,reason="совпадение геометрии или неподтверждённый сценарий")
            todo,attempt = retry,attempt+1
            if attempt>=1000:
                raise RuntimeError("Исчерпаны зёрна геометрии")
        db.commit()
        if pos%300==0 or pos+30>=len(needed):
            done = len(existing)+min(pos+30,len(needed))
            log("corpus_progress",split=split,geometries=done,certified=4*done)
    if needed:
        log("corpus_milestone",split=split,geometries=target,certified=4*target,seconds=round(time.perf_counter()-start,2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    selfcheck()
    excluded={geometry(c.level.grid) for cases in development_sets() for c in cases}
    excluded.update(r["geometry"] for r in json.loads((DATA/"test-manifest.json").read_text()).values())
    schedule = [("validation_id",60),("train",300),("train",3000),("train",30000),("train",60000),
                ("validation_id",3000),("shift_development",1500),("test_id",6000),("test_ood",6000)]
    with connect() as db,ProcessPoolExecutor(args.workers) as pool:
        for split,n in schedule:
            build(db,pool,split,n,excluded)


if __name__=="__main__":
    main()
