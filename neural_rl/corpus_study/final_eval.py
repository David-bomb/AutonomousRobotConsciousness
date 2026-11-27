"""Отбор по val, фиксация весов и покартовая оценка на закрытых выборках."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
import time

from .session import ROOT,OUT,check_time,log
from .report import NAMES,load_runs,table,write_text

BLOCK=100
SIZES={"validation_id":3000,"shift_development":1500,"test_id":6000,"test_ood":6000}
SPLIT_NAMES={"validation_id":"Выбор модели","shift_development":"Умеренные изменения",
             "test_id":"Новые карты привычных типов","test_ood":"Новые условия"}
GROUP_NAMES={"size":"Увеличенный размер","generator":"Другой генератор","topology":"Изменённые связи дорог",
             "services":"Размещение сервисов","resources":"Начальные запасы","combined":"Несколько изменений"}


def wait_for_split(split):
    from .corpus import DB
    with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
        while True:
            check_time()
            count,=db.execute("SELECT count(*) FROM geoms WHERE split=?",(split,)).fetchone()
            if count==SIZES[split]:
                return
            time.sleep(10)


def mean(xs):
    return statistics.mean(xs)


def rank(stats,seed):
    length=stats["mean_success_steps"]
    return stats["worst"],stats["wins"],-(float("inf") if length is None else length),-seed


def shortlist(runs):
    if len(runs)!=200 or len({r["config"]["id"] for r in runs})!=200:
        raise ValueError("Отбор начнётся после 200 завершённых обучений")
    selected={}
    for family in NAMES:
        configs=defaultdict(list)
        for r in runs:
            if r["config"]["family"]==family:
                configs[r["config"]["variant"]].append(r)
        if len(configs)!=10 or any(len(group)!=5 for group in configs.values()):
            raise RuntimeError("Неполная матрица семейства")
        choices=sorted(configs,key=lambda v:(mean(r["screen"]["worst"] for r in configs[v]),
                                              mean(r["screen"]["wins"] for r in configs[v]),-v),reverse=True)[:2]
        selected[family]=[max(configs[v],key=lambda r:rank(r["screen"],r["config"]["seed"]))["config"]["id"]
                          for v in choices]
    return selected


def freeze_file(path,record):
    from neural_rl.experiment import save,read
    if path.exists():
        if read(path)!=record: raise RuntimeError(f"Не совпала фиксация {path}")
    else:
        save(path,record)


def score(model_id,split):
    from neural_rl.experiment import digest,read,save
    from .learn import DirectNetwork,Pool,evaluate_rows,metrics,setup
    import sqlite3
    import torch
    from .corpus import DB
    run_dir=OUT/"runs"/model_id
    run=read(run_dir/"run.json")
    config=run["config"]
    weight_hash=digest(run_dir/"weights.pt")
    if weight_hash!=run["weights_sha256"]:
        raise RuntimeError("Не совпали веса")
    if split!="validation_id":
        frozen=read(OUT/"frozen.json")
        if not any(v["id"]==model_id and v["sha256"]==weight_hash for v in frozen.values()):
            raise RuntimeError("Модель не закреплена для итогового контроля")
    folder=OUT/"evaluation"/split/model_id
    folder.mkdir(parents=True,exist_ok=True)
    setup(config["seed"])
    net=DirectNetwork(config).cuda()
    net.load_state_dict(torch.load(run_dir/"weights.pt",weights_only=True,map_location="cuda"))
    rows=[]
    groups=defaultdict(list)
    for start in range(0,SIZES[split],BLOCK):
        check_time()
        stop=min(SIZES[split],start+BLOCK)
        path=folder/f"{start:05}.json.gz"
        if not path.exists():
            pool=Pool(split,stop-start,offset=start)
            labels={i:g for i,g in pool.db.execute(
                "SELECT idx,group_name FROM geoms WHERE split=? AND idx>=? AND idx<?",(split,start,stop))}
            part=evaluate_rows(net,pool,chunk=64)
            for r in part:
                r["group"]=labels[int(r["map"].rsplit("-",2)[1])]
            pool.db.close()
            pending=path.with_suffix(".pending")
            with gzip.open(pending,"wt",encoding="utf-8") as stream:
                json.dump(dict(weights_sha256=weight_hash,rows=part),stream,separators=(",",":"))
            pending.replace(path)
            if start%500==0 or stop==SIZES[split]:
                log("evaluation_progress",id=model_id,split=split,scenarios=stop*4)
        with gzip.open(path,"rt",encoding="utf-8") as stream: saved=json.load(stream)
        if saved["weights_sha256"]!=weight_hash or len(saved["rows"])!=4*(stop-start):
            raise ValueError("Нарушена целостность части оценки")
        rows.extend(saved["rows"])
    if len({r["map"] for r in rows})!=len(rows): raise ValueError("Повтор сценария")
    for r in rows: groups[r["group"]].append(r)
    result=dict(id=model_id,split=split,weights_sha256=weight_hash,**metrics(rows),
                groups={k:metrics(v) for k,v in groups.items()})
    save(folder/"summary.json",result)
    log("evaluation_completed",id=model_id,split=split,wins=result["wins"],scenarios=len(rows))


def launch(job):
    model,split=job
    check_time()
    path=OUT/"process-logs"/f"eval-{model}-{split}.txt"
    with path.open("ab") as stream:
        return subprocess.call([sys.executable,"-B","-u","-m","neural_rl.corpus_study.final_eval",
                                "score","--model",model,"--split",split],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)


def jobs(items):
    with ThreadPoolExecutor(2) as pool:
        codes=list(pool.map(launch,items))
    for job,code in zip(items,codes):
        if code:
            log("evaluation_retry",id=job[0],split=job[1],parallel=1)
            if launch(job): raise RuntimeError(f"Требуется разбор оценки {job}")


def paired_lengths(frozen,split):
    successes={}
    for family,record in frozen.items():
        by_map={}
        folder=OUT/"evaluation"/split/record["id"]
        for fn in sorted(folder.glob("*.json.gz")):
            with gzip.open(fn,"rt",encoding="utf-8") as stream:
                rows=json.load(stream)["rows"]
            by_map.update({r["map"]:r["steps"] for r in rows if r["outcome"]=="SUCCESS"})
        successes[family]=by_map
    pairs=[]
    for left,right in combinations(frozen,2):
        common=successes[left].keys() & successes[right].keys()
        both=[round(mean(successes[f][m] for m in common),2) if common else "—" for f in (left,right)]
        pairs.append([SPLIT_NAMES[split],NAMES[left],NAMES[right],len(common),*both])
    return pairs


def main():
    from neural_rl.experiment import read
    runs=load_runs()
    choices=shortlist(runs)
    freeze_file(OUT/"shortlist.json",choices)
    wait_for_split("validation_id")
    jobs([(i,"validation_id") for pair in choices.values() for i in pair])
    frozen={}
    for f,pair in choices.items():
        def key(i):
            stats=read(OUT/"evaluation"/"validation_id"/i/"summary.json")
            c=read(OUT/"runs"/i/"run.json")["config"]
            return *rank(stats,c["seed"])[:3],-c["variant"],-c["seed"]
        chosen=max(pair,key=key)
        frozen[f]=dict(id=chosen,sha256=read(OUT/"runs"/chosen/"run.json")["weights_sha256"])
    freeze_file(OUT/"frozen.json",frozen)
    log("models_frozen",families=frozen)
    for split in ("shift_development","test_id","test_ood"):
        wait_for_split(split)
        jobs([(v["id"],split) for v in frozen.values()])
    results=[]
    details=[]
    failures=[]
    difficulty=[]
    for f,v in frozen.items():
        scores=[read(OUT/"evaluation"/split/v["id"]/"summary.json") for split in SIZES]
        results.append([NAMES[f],v["id"],*[f'{s["wins"]}/{s["scenarios"]}' for s in scores]])
        for s in scores:
            outcomes=s["outcomes"]
            failures.append([NAMES[f],SPLIT_NAMES[s["split"]],s["wins"],outcomes.get("EMPTY",0),
                             outcomes.get("BROKEN",0),outcomes.get("LIMIT",0)])
            difficulty.append([NAMES[f],SPLIT_NAMES[s["split"]],*[
                f'{100*s["by_stage"][str(k)]:.2f}%' if str(k) in s["by_stage"] else "—"
                for k in (1,2,3)]])
        for g,s in scores[-1]["groups"].items():
            details.append([NAMES[f],GROUP_NAMES[g],s["wins"],s["scenarios"],s["outcomes"].get("EMPTY",0),s["outcomes"].get("BROKEN",0)])
    write_text(OUT/"final-results.md","## Сколько лабиринтов прошли выбранные модели\n\n"
               "В каждой ячейке указаны финиши и число поездок. По полной проверочной выборке выбраны веса "
               "одной модели каждого семейства. Три правых столбца показывают её результат после фиксации весов.\n\n"+
               table(["Семейство","Выбранная модель",*SPLIT_NAMES.values()],results)+
               "\n\n### Исходы поездок выбранных сетей\n\n"+
               table(["Семейство","Набор","Финиш","Кончилось топливо","Поломка","Лимит"],failures)+
               "\n\n### Финиши по размеру карты\n\nДоля финишей на каждой ступени, определяемой длиной большей стороны карты.\n\n"+
               table(["Семейство","Набор","До 15","16–21","От 22"],difficulty)+
               "\n\n### Что изменилось в отложенных лабиринтах\n\n"+table(["Семейство","Условие","Финиши","Сценарии","Топливо","Поломки"],details)+
               "\n\n### Число ходов на общих финишах\n\n"+
               table(["Набор","Первая сеть","Вторая сеть","Общих финишей","Ходов первой","Ходов второй"],
                     paired_lengths(frozen,"test_id")+paired_lengths(frozen,"test_ood"))+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("score","all"))
    parser.add_argument("--model")
    parser.add_argument("--split",choices=SIZES)
    args=parser.parse_args()
    if args.action=="score": score(args.model,args.split)
    else: main()
