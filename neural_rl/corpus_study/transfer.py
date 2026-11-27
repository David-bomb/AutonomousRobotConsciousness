"""Заранее заданные умеренные сдвиги для shift_development."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import random
import sqlite3
import zlib

from maze.datasets import geometry,development_sets,DATA
from maze.levels import Level
from .corpus import DB, MOVES, certify, design, edges, scenario
from .geometry import generate
from .session import OUT,check_time,log
from .report import write_text


def vary(rows,group,seed):
    rng=random.Random(seed)
    for row in rows:
        title,stocks,*grid=row["source"].splitlines()
        level=Level.parse(row["source"])
        fuel,health=map(int,stocks.split())
        if group in ("services","combined") and rng.random()<.30:
            symbol=rng.choice("FR")
            stations=[(x,y) for y,line in enumerate(grid) for x,c in enumerate(line) if c==symbol]
            if stations:
                a,b=rng.choice(stations)
                options=[(x,y) for y,line in enumerate(grid) for x,c in enumerate(line) if c=="." and
                         1<=max(abs(x-a),abs(y-b))<=8]
                if options:
                    x,y=rng.choice(options)
                    grid[y]=grid[y][:x]+symbol+grid[y][x+1:]
        if group in ("resources","combined"):
            x,y=level.start
            first={}
            for step,code in enumerate(row["moves"],1):
                dx,dy=MOVES[int(code)].value
                x,y=x+dx,y+dy
                first.setdefault(grid[y][x],step)
            fuel=min(fuel,first.get("F",len(row["moves"]))+rng.randint(3,8))
            health=min(health,first.get("R",len(row["moves"]))+rng.randint(3,8))
        source="\n".join([title,f"{fuel} {health}",*grid])
        proof=certify(Level.parse(source),row["moves"])
        if not proof: raise ValueError("Умеренный сдвиг не прошёл физический повтор")
        row.update(source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),proof=proof)


def make(task):
    i,attempt=task
    check_time()
    split="shift_development"
    family,size,group=design(split,i)
    seed=8*10**11+i*1000+attempt
    rng=random.Random(seed)
    varied=group in ("parameters","combined")
    grid=generate(family,size,rng,sparse=varied,height=size+4 if varied else size)
    if varied and family in ("rooms","field"):
        free=[(x,y) for y,line in enumerate(grid) for x,c in enumerate(line) if c=="."]
        x,y=rng.choice(free)
        grid=list(grid)
        for j in range(y,min(len(grid)-1,y+5)):
            end=min(len(grid[j])-1,x+7)
            grid[j]=grid[j][:x]+"."*(end-x)+grid[j][end:]
        grid=tuple(grid)
    graph=edges(grid)
    try:
        rows=[scenario(grid,graph,j,rng,f"{split}-{i:05}-{j}","standard") for j in range(4)]
        vary(rows,group,seed+5)
    except (ValueError,IndexError) as exc:
        return str(exc)
    return split,i,geometry(grid),family,max(size,len(grid)),seed,group,zlib.compress(json.dumps(rows,separators=(",",":")).encode(),3)


def main():
    digest=hashlib.sha256(open(__file__,"rb").read()).hexdigest()
    method=dict(source_sha256=digest,split="shift_development",geometries=1500,groups=5,
                parameters="Вытянутые прямоугольники, меньше поперечных связей у braid, локальная комната у rooms и field.",
                resources="Начальный ресурс ограничен длиной подтверждённого хода до первого обслуживания +3…8 или длиной до финиша +3…8.",
                services="В 30% сценариев дополнительная станция располагается не далее восьми клеток по координатам от станции того же типа.")
    path=OUT/"shift-method.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=method:
        raise RuntimeError("Рецепт уже зарегистрирован")
    write_text(path,json.dumps(method,ensure_ascii=False,indent=2)+"\n")
    log("shift_design_registered",**method)
    excluded={geometry(c.level.grid) for cases in development_sets() for c in cases}
    excluded.update(r["geometry"] for r in json.loads((DATA/"test-manifest.json").read_text()).values())
    with sqlite3.connect(DB,timeout=90) as db,ProcessPoolExecutor(2) as pool:
        existing={r[0] for r in db.execute("SELECT idx FROM geoms WHERE split='shift_development'")}
        for start in range(0,1500,30):
            retry=[i for i in range(start,start+30) if i not in existing]
            attempt=0
            while retry:
                tasks=[(i,attempt) for i in retry]
                results=list(pool.map(make,tasks,chunksize=2))
                again=[]
                with db:
                    for i,result in zip(retry,results):
                        if isinstance(result,str) or result[2] in excluded:
                            again.append(i)
                            continue
                        try: db.execute("INSERT INTO geoms VALUES (?,?,?,?,?,?,?,?)",result)
                        except sqlite3.IntegrityError: again.append(i)
                retry,attempt=again,attempt+1
                if attempt>=1000: raise RuntimeError("Нет уникальной геометрии")
            if start%300==0: log("shift_progress",geometries=start+30)
    log("corpus_milestone",split="shift_development",geometries=1500,certified=6000,recipe=digest)


if __name__=="__main__": main()
