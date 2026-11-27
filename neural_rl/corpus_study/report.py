"""Сводка серии строится по подтверждённым записям и завершённым обучениям."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics

from .session import OUT

NAMES={"double":"Double DQN + CNN","dueling":"Dueling Double DQN","rainbow":"Rainbow DQN","ppo":"PPO + CNN"}


def load_runs():
    items=[]
    for p in sorted((OUT/"runs").glob("*/run.json")):
        row=json.loads(p.read_text(encoding="utf-8"))
        if p.parent.name!=row["config"]["id"]:
            raise ValueError("Нарушена привязка результата к запуску")
        items.append(row)
    return items


def write_text(path,text):
    tmp=path.with_suffix(f".{os.getpid()}.writing")
    tmp.write_text(text,encoding="utf-8")
    tmp.replace(path)


def table(columns,rows):
    data=[columns,["---"]*len(columns),*rows]
    return "\n".join("| "+" | ".join(map(str,r))+" |" for r in data)


def human_journal():
    rows=[]
    with (OUT/"experiments.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"): continue
            event=json.loads(line)
            kind=event["event"]
            if kind=="trial_started":
                c=event["config"]
                label="Начато техническое обучение" if event["technical"] else "Начато обучение"
                detail=f'{c["geometry_count"]} геометрий, зерно {c["seed"]}, процесс {event["pid"]}'
            elif kind=="trial_completed":
                label="Техническая проверка завершена" if event["technical"] else "Обучение завершено"
                detail=(f'{event["train_steps"]} ходов, {event["updates"]} обновлений. '
                        f'Финишей до и после {event["initial"]} → {event["wins"]} из {event["val"]}. '
                        f'{event["seconds"]} с.')
            elif kind=="corpus_milestone":
                label="Допущен набор "+event["split"]
                detail=f'{event["geometries"]} геометрий, {event["certified"]} сценариев'
            elif kind in ("trial_failure","trial_needs_repair","gpu_fallback","evaluation_completed",
                          "models_frozen","witness_audit_completed","diagnostics_certified",
                          "session_started","gpu_concurrency_changed","session_finished","diagnostic_training_registered",
                          "diagnostic_training_started","diagnostic_training_completed","diagnostic_trained_control","final_verification"):
                label=kind
                detail=json.dumps({k:v for k,v in event.items() if k not in ("time","event","id","detail")},ensure_ascii=False)
            else:
                continue
            stamp=datetime.fromisoformat(event["time"]).astimezone(timezone(timedelta(hours=3))).strftime("%d.%m %H:%M:%S")
            rows.append([stamp,event.get("id","—"),label,detail.replace("|","/")])
    write_text(OUT/"journal.md","# Журнал экспериментов E-0012\n\nСобытия по московскому времени. Для каждого старта и завершения указана конфигурация. "
               "Технические проверки отделены от основной серии. Подробности и повторы генерации сохранены в [полном журнале](experiments.jsonl).\n\n"+
               table(["Время","Запуск","Событие","Результат или условия"],rows)+"\n")


def generate():
    runs=load_runs()
    db=sqlite3.connect((OUT/"corpus.sqlite").as_uri()+"?mode=ro",uri=True)
    counts=dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))
    checked=sum(counts.values())*4
    diagnostics=OUT/"diagnostics"/"verified.json"
    diag=json.loads(diagnostics.read_text(encoding="utf-8")) if diagnostics.exists() else {}
    extra=diag.get("verified_episodes",0)
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime("%d.%m.%Y %H:%M:%S")
    out=["# E-0012. Ночная серия прямого RL",f"Срез на {stamp} по Москве. "
         f"Подтверждённых сценариев основного корпуса — **{checked:,}**, дополнительных диагностических — **{extra:,}**. "
         f"Завершено **{len(runs)} из 200 обучений**. Каждая учтённая модель прошла обучение на CUDA и общую промежуточную проверку.",
         "## Подтверждение проходимости",
         table(["Набор","Геометрий","Сценариев с финишем в симуляторе"],
               [(s,counts.get(s,0),4*counts.get(s,0)) for s in
                ("train","validation_id","shift_development","test_id","test_ood")]),
         "В каждой записи корпуса сохранены исходные запасы, уровень, последовательность ходов и результат её исполнения через Environment.step. Сеть получает штатное наблюдение. Все четыре сценария одной геометрии принадлежат одной выборке.",
         "## Промежуточная проверка по семействам",
         "Все запуски проходят одни и те же 240 сценариев validation_id до и после обучения. В столбцах до и после указано среднее число финишей из 240. Причины остановки суммируются по завершённым моделям."]
    familyrows=[]
    for f,title in NAMES.items():
        group=[r for r in runs if r["config"]["family"]==f]
        if not group:
            familyrows.append([title,0,"—","—","—","—","—","—"])
            continue
        failures=Counter()
        for r in group: failures.update(r["screen"]["outcomes"])
        familyrows.append([title,len(group),f'{statistics.mean(r["initial"]["wins"] for r in group):.2f}',
                           f'{statistics.mean(r["screen"]["wins"] for r in group):.2f}',
                           max(r["screen"]["wins"] for r in group),failures["EMPTY"],failures["BROKEN"],failures["LIMIT"]])
    out.append(table(["Семейство","Обучений","До","После","Максимум","Топливо","Поломка","Лимит"],familyrows))
    completed=OUT/"session-result.json"
    if completed.is_file():
        session=json.loads(completed.read_text(encoding="utf-8"))
        out[1]=(f"Серия завершена. Проверены **{checked:,} основных** и **{extra:,} диагностических сценариев**. "
                f"Выполнены **{len(runs)} независимых обучений** и {session['diagnostic_continuations']} диагностических "
                f"продолжения от выбранных весов. Итоговая сверка завершилась через "
                f"{session['verification_seconds']//3600} ч {session['verification_seconds']%3600//60} мин. "
                f"Одновременно обучались до {session['max_parallel_trainings']} моделей на CUDA. "
                "[Сведения о сессии](session-result.json).")
    out.append(f"Всего физических шагов обучения — {sum(r['train_steps'] for r in runs):,}. "
               f"Обновлений весов — {sum(r['updates'] for r in runs):,}. Технические прогоны записаны отдельно в smoke.")
    out.extend(["## Размер корпуса и изменения конфигурации",
                "Для каждой строки указано среднее число финишей из 240 на промежуточной val. В скобках — минимум и максимум по инициализациям, затем число обучений. "
                "Варианты 1–4 различаются объёмом доступного train. Остальные сравниваются с вариантом 2 на тех же 3 000 геометрий."])
    labels=("300 геометрий","3 000 геометрий","30 000 геометрий","60 000 геометрий",
            "Две свёртки","16 каналов","Шаг Adam 0,0003","Дисконтирование 0,99",
            "Обновление целевой сети / отсечение PPO","Порция Q / энтропия PPO")
    variants=[]
    for v,label in enumerate(labels,1):
        parts=[]
        for family in NAMES:
            group=[r for r in runs if r["config"]["variant"]==v and r["config"]["family"]==family]
            wins=[r["screen"]["wins"] for r in group]
            parts.append(f'{statistics.mean(wins):.2f} ({min(wins)}–{max(wins)}, n={len(wins)})' if wins else "—")
        variants.append([v,label,*parts])
    out.append(table(["№","Условие",*NAMES.values()],variants))
    if runs:
        coverage=[]
        for v in range(1,5):
            group=[r for r in runs if r["config"]["variant"]==v]
            if group:
                coverage.append([group[0]["config"]["geometry_count"],len(group),
                                 min(r["train_geometries"] for r in group),max(r["train_geometries"] for r in group),
                                 min(r["train_scenarios"] for r in group),max(r["train_scenarios"] for r in group)])
        out.extend(["### Фактический охват при равном числе физических шагов",
                    table(["Геометрий в банке","Опорных обучений","Геометрий, мин.","Геометрий, макс.","Сценариев, мин.","Сценариев, макс."],coverage)])
    rows=[]
    for r in runs:
        c=r["config"]; screen=r["screen"]
        row=dict(id=c["id"],family=c["family"],train_set=c["geometry_count"],variant=c["variant"],seed=c["seed"],
                 steps=r["train_steps"],updates=r["updates"],geometries_seen=r["train_geometries"],scenarios_seen=r["train_scenarios"],
                 before=r["initial"]["wins"],after=screen["wins"],gain=screen["wins"]-r["initial"]["wins"],
                 worst=screen["worst"],mean_success_steps=screen["mean_success_steps"],gpu_mib=r["peak_gpu_mib"],
                 seconds=r["seconds"],weights_sha256=r["weights_sha256"])
        rows.append(row)
    if rows:
        tmp=OUT/f"runs.{os.getpid()}.writing"
        with tmp.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
        tmp.replace(OUT/"runs.csv")
    final_results=OUT/"final-results.md"
    if final_results.exists():
        headline, separator, details=final_results.read_text(encoding="utf-8").partition("\n\n### Исходы поездок выбранных сетей")
        out.insert(2,headline.strip())
        if separator:
            out.append("## Исходы поездок выбранных сетей"+details)
    for section in ("diagnostic-results.md","diagnostics/training-results.md","audit.md","verification.md"):
        if (OUT/section).exists():
            out.append((OUT/section).read_text(encoding="utf-8").strip())
    if (OUT/"corpus-profile.md").exists():
        out.append("Для каждого вложенного набора сверены шесть типов геометрий и пять размеров. "
                   "[Состав выборок и повтор генерации из сохранённых зёрен](corpus-profile.md).")
    human_journal()
    out.extend(["## Файлы серии",
                "[Регистрация 200 обучений](registration.json) · [Хронология обучений](journal.md) · [Все эксперименты и события](experiments.jsonl) · "
                "[Сводка запусков](runs.csv) · [Картотека с маршрутами для повтора](corpus.sqlite)."])
    write_text(OUT/"report.md","\n\n".join(out)+"\n")
    print(f"certified={checked+extra} main={checked} extra={extra} runs={len(runs)}",flush=True)


if __name__=="__main__":
    generate()
