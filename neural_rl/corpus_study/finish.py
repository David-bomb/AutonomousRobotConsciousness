"""Обновление ночного отчёта и переход к закрытым испытаниям после обучения."""
import json
import sqlite3
import subprocess
import sys
import time

from .corpus import DB,TARGETS
from .report import generate
from .session import DEADLINE,OUT,ROOT,locked,log


def start(label,module,*args):
    stream=(OUT/"process-logs"/f"{label}.txt").open("ab")
    try:
        proc=subprocess.Popen([sys.executable,"-B","-u","-m","neural_rl.corpus_study."+module,*args],
                              cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        stream.close()
    log("phase_started",phase=label,pid=proc.pid)
    return proc


def run():
    children={}
    completed=set()
    next_report=0.
    while True:
        now=time.time()
        for name,proc in list(children.items()):
            code=proc.poll()
            if code is not None:
                log("phase_exited",phase=name,code=code)
                del children[name]
                if code:
                    generate()
                    raise RuntimeError(f"{name}: {code}, см. process-logs")
                completed.add(name)
        if now>=next_report:
            generate()
            next_report=now+90
        if now>=DEADLINE:
            if children:
                time.sleep(2)
                continue
            generate()
            return
        with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
            counts=dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))
        if counts==TARGETS and "full-audit" not in children and "full-audit" not in completed:
            children["full-audit"]=start("full-audit","audit","--workers","2")
        ready=(len(list((OUT/"runs").glob("*/run.json")))==200
               and counts.get("validation_id")==TARGETS["validation_id"])
        if ready and "final-evaluation" not in children and "final-evaluation" not in completed:
            children["final-evaluation"]=start("final-evaluation","final_eval","all")
        if "final-evaluation" in completed and "diagnostic-evaluation" not in children and "diagnostic-evaluation" not in completed:
            children["diagnostic-evaluation"]=start("diagnostic-evaluation","diagnostic_eval")
        if len(completed)==3:
            generate()
            return
        time.sleep(10)


if __name__=="__main__":
    with locked(OUT/".finish.lock"):
        run()
