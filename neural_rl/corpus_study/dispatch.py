"""Регистрация и параллельный запуск двух независимых обучений на одном GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time

from .session import ROOT, OUT, DEADLINE, check_time, log, locked


def register(ticks):
    import torch
    import numpy as np
    from .learn import FAMILIES,config_for
    builds = [config_for(f,v,s,ticks) for v in range(1,11) for s in range(1,6) for f in FAMILIES]
    measurements = {}
    for f in FAMILIES:
        result=json.loads((OUT/"smoke"/f"{f}-v01-s1"/"run.json").read_text(encoding="utf-8"))
        measurements[f]={k:result[k] for k in ("train_steps","train_seconds","peak_gpu_mib","updates","parameter_delta")}
    source_files = [*sorted((ROOT/"neural_rl").glob("*.py")),
                    *(ROOT/"neural_rl/corpus_study"/p for p in ("learn.py","geometry.py","corpus.py","session.py")),
                    *(ROOT/"maze"/p for p in ("environment.py","memory.py","grid.py","levels.py","interface.py","datasets.py"))]
    record = dict(runs=builds,primary_train_steps=3*24*ticks,screen_geometry=60,screen_scenarios=240,
                  primary_screen="Первые 60 заранее созданных независимых геометрий validation_id. Все 240 сценариев до и после.",
                  finalists="В каждом семействе две лучшие конфигурации по среднему пяти зёрен: худшая доля по сложности, общая доля, меньший номер. Представитель каждой — по худшей доле, числу финишей, меньшему числу ходов на финишах, меньшему зерну. Победителя выбирает полная validation_id на 12000 сценариев по тем же показателям. После фиксации — закрытые выборки.",
                  schedules="Перестановка учебных сценариев внутри ступени по размеру: до 15, 21, от 31. На последующих ступенях добавлены 10 сценариев прежних сложностей, как в E-0011.",
                  input_canvas=81,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  smoke=measurements,python=platform.python_version(),torch=str(torch.__version__),
                  numpy=np.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    manifest=OUT/"registration.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))!=record:
            raise ValueError("Конфликт регистрации")
        return
    from neural_rl.experiment import save
    save(manifest,record)
    log("registered",runs=len(builds),steps=record["primary_train_steps"],parallel=2,smoke=measurements)


def run():
    settings=json.loads((OUT/"registration.json").read_text(encoding="utf-8"))
    queue=[]
    for c in settings["runs"]:
        path=OUT/"runs"/c["id"]/"run.json"
        if path.is_file():
            row=json.loads(path.read_text(encoding="utf-8"))
            if row["config"]!=c or hashlib.sha256((path.parent/"weights.pt").read_bytes()).hexdigest()!=row["weights_sha256"]:
                raise ValueError(f"Нарушена целостность ранее завершённого {c['id']}")
        else:
            queue.append(c)
    limit=2
    active={}
    attempts={}
    errors=OUT/"process-logs"
    errors.mkdir(exist_ok=True)
    db=sqlite3.connect((OUT/"corpus.sqlite").as_uri()+"?mode=ro",uri=True)
    while queue or active:
        for key,(p,handle,c) in list(active.items()):
            if p.poll() is None:
                continue
            handle.close()
            del active[key]
            if p.returncode:
                if time.time()>=DEADLINE:
                    log("deadline_trial",id=key)
                    continue
                detail=(errors/f"{key}.txt").read_text(encoding="utf-8",errors="replace")[-4000:]
                shortage=any(term in detail.lower() for term in
                             ("out of memory","memoryerror","alloc_failed","paging file","1455","недостаточ","bad allocation"))
                log("trial_failure",id=key,returncode=p.returncode,resource=shortage,detail=detail)
                if shortage:
                    limit=1
                    log("gpu_fallback",parallel=1,id=key)
                attempts[key]=attempts.get(key,0)+1
                if attempts[key]<=2:
                    queue.insert(0,c)
                else:
                    # Продолжить независимые обучения. Причина останется в журнале для оператора.
                    log("trial_needs_repair",id=key)
        if time.time()>=DEADLINE:
            if not active:
                break
            time.sleep(.5)
            continue
        if queue and len(active)<limit:
            count,=db.execute("SELECT count(*) FROM geoms WHERE split='train'").fetchone()
            ready=next((c for c in queue if c["geometry_count"]<=count),None)
            if ready:
                c=ready
                queue.remove(c)
                key=c["id"]
                stdout=(errors/f"{key}.txt").open("ab")
                seed_no=int(key.rsplit("s",1)[1])
                args=[sys.executable,"-B","-u","-m","neural_rl.corpus_study.learn",c["family"],str(c["variant"]),str(seed_no)]
                popen=subprocess.Popen(args,stdout=stdout,stderr=subprocess.STDOUT,cwd=ROOT)
                active[key]=(popen,stdout,c)
                continue
        if queue or active:
            time.sleep(1.5)
    log("dispatcher_stopped",remaining=len(queue),deadline=time.time()>=DEADLINE)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("register","run"))
    parser.add_argument("--ticks",type=int,default=1024)
    args=parser.parse_args()
    check_time()
    if args.action=="register":
        register(args.ticks)
    else:
        with locked(OUT/".dispatcher.lock"):
            run()
