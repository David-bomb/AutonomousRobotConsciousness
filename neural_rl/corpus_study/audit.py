"""Независимое чтение базы и пошаговое повторение маршрутов допуска."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import sqlite3
import zlib

from maze.datasets import DATA,development_sets,geometry
from maze.environment import Environment,Status
from maze.interface import Move
from maze.levels import Level
from .corpus import DB,SPLITS,TARGETS
from .session import OUT,check_time,log
from .report import table,write_text


def replay(record,expected_name):
    source=record["source"]
    if hashlib.sha256(source.encode()).hexdigest()!=record["sha256"]:
        raise ValueError("Отпечаток сценария "+expected_name)
    level=Level.parse(source)
    if level.title!=expected_name: raise ValueError("Имя сценария "+expected_name)
    env=Environment(level)
    directions=tuple(Move)
    for action in record["moves"]:
        env.step(directions[int(action)])
        if not env.view.moved: raise ValueError("Невыполненный ход "+expected_name)
    proof=dict(status=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
               final_fuel=env.fuel,final_health=env.health)
    if env.status!=Status.SUCCESS or proof!=record["proof"]: raise ValueError("Не совпало исполнение "+expected_name)
    return level,env


def block(task):
    split,begin,end=task
    check_time()
    with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
        rows=db.execute("SELECT idx,hash,data FROM geoms WHERE split=? AND idx>=? AND idx<? ORDER BY idx",
                        (split,begin,end)).fetchall()
    if len(rows)!=end-begin: raise ValueError("Неполная часть корпуса")
    totals=Counter()
    for i,key,data in rows:
        check_time()
        items=json.loads(zlib.decompress(data))
        if len(items)!=4: raise ValueError("Неполная геометрия")
        for j,record in enumerate(items):
            level,env=replay(record,f"{split}-{i:05}-{j}")
            if geometry(level.grid)!=key: raise ValueError("Не совпал рисунок стен")
            totals["scenarios"]+=1
            totals["steps"]+=env.steps
            totals["required_fuel_lower_bound"]+=level.fuel<record["direct"]
            totals["required_health_lower_bound"]+=level.health<record["direct"]
            totals["refuels"]+=env.refuels
            totals["repairs"]+=env.repairs
    return split,begin,end,dict(totals)


def main(workers):
    out=OUT/"audit"
    out.mkdir(exist_ok=True)
    with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()!=("ok",): raise ValueError("База повреждена")
        counts=dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))
        new_hashes={r[0] for r in db.execute("SELECT hash FROM geoms")}
    old_hashes={geometry(c.level.grid) for group in development_sets() for c in group}
    old_hashes.update(r["geometry"] for r in json.loads((DATA/"test-manifest.json").read_text()).values())
    if new_hashes & old_hashes or len(new_hashes)!=sum(counts.values()):
        raise ValueError("Повтор геометрии между выборками")
    tasks=[(split,i,min(i+100,n)) for split,n in counts.items() for i in range(0,n,100)]
    def filename(task):
        return out/f"{task[0]}-{task[1]:05}-{task[2]:05}.json"
    pending=[t for t in tasks if not filename(t).exists()]
    with ProcessPoolExecutor(workers) as pool:
        for split,begin,end,totals in pool.map(block,pending,chunksize=1):
            task=(split,begin,end)
            values=dict(split=split,begin=begin,end=end,**totals)
            write_text(filename(task),json.dumps(values)+"\n")
            if end%5000==0 or end==counts[split]: log("witness_audit_progress",split=split,verified_through=end)
    totals={s:Counter() for s in counts}
    for t in tasks:
        data=json.loads(filename(t).read_text())
        totals[t[0]].update({k:v for k,v in data.items() if k not in ("split","begin","end")})
    rows=[[s,counts[s],totals[s]["scenarios"],totals[s]["steps"],totals[s]["required_fuel_lower_bound"],
           totals[s]["required_health_lower_bound"]] for s in SPLITS if s in counts]
    write_text(OUT/"audit-summary.json",json.dumps(dict(all_splits=counts==TARGETS,
               geometries=counts,totals=totals),ensure_ascii=False,indent=2)+"\n")
    write_text(OUT/"audit.md","## Повторное подтверждение проходимости\n\n"+
               table(["Набор","Геометрий","Повторных финишей","Физических ходов","Требуется заправка¹","Требуется ремонт¹"],rows)+
               "\n\n¹ Нижняя оценка числа сценариев, где начальный ресурс меньше длины кратчайшего пути до финиша. "
               "Для каждой записи независимо прочитаны карта и команды, проверена контрольная сумма и повторены физические ходы до финиша.\n")
    log("witness_audit_completed",snapshot=counts,verified=sum(r["scenarios"] for r in totals.values()),
        all_splits=counts==TARGETS)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    main(parser.parse_args().workers)
