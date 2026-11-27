"""Сверка покартовых исходов и отчёт по эксперименту E-0013."""
from collections import Counter
import csv
import json
from statistics import mean

import torch

from maze.environment import Environment, Status
from neural_rl import perception
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.stress_hundreds import MOVES, OUT, ROOT, SOURCE, SPLITS, maps
from neural_rl.corpus_study.report import NAMES, table, write_text


@torch.no_grad()
def repeat(weights, config, rows, levels, include_failures):
    net=Network(config).cuda().eval()
    net.load_state_dict(torch.load(weights,map_location="cuda",weights_only=True))
    samples={}
    for row in rows:
        if row["outcome"]=="SUCCESS":
            samples[row["map"]]=row
            continue
        if not include_failures:
            continue
        key=row["split"],row["outcome"]
        if key not in samples or (row["steps"],row["map"]) < (samples[key]["steps"],samples[key]["map"]):
            samples[key]=row
    active=[(r,Environment(levels[r["map"]])) for r in samples.values()]
    winning_paths={r["map"]:"" for r in samples.values() if r["outcome"]=="SUCCESS"}
    while active:
        actions=net(batch([perception.encode(env.view,False) for _,env in active]))[0].argmax(1).tolist()
        following=[]
        for (r,env),a in zip(active,actions):
            env.step(MOVES[a])
            if r["map"] in winning_paths:
                winning_paths[r["map"]]+=str(a)
            if env.status==Status.RUNNING:
                following.append((r,env))
            else:
                assert (env.status.name,env.steps,env.refuels,env.repairs)==tuple(r[k] for k in ("outcome","steps","refuels","repairs")),r["map"]
        active=following
    return [r["map"] for r in samples.values()],winning_paths


def main():
    spec=read(OUT/"manifest.json")
    models=spec["models"]
    frozen=read(SOURCE/"frozen.json")
    cases=maps()
    reference={c.name:s for s,c in cases}
    levels={c.name:c.level for _,c in cases}
    proofs=read(OUT/"witnesses.json")
    assert set(proofs)==set(levels) and len(levels)==300
    require_both=Counter()
    for name,proof in proofs.items():
        env=Environment(levels[name])
        direct=len(perception.astar(env.level.grid,env.position,env.level.goal))
        require_both[reference[name]]+=max(env.fuel,env.health)<direct
        for code in proof["actions"]:
            env.step(MOVES[int(code)])
            assert env.moved
        assert env.status==Status.SUCCESS and (env.steps,env.refuels,env.repairs)==tuple(proof[k] for k in ("steps","refuels","repairs"))
    assert {p.stem for p in (OUT/"models").glob("*.json")}==set(models)
    scores=[]
    for name,m in models.items():
        point=read(OUT/"models"/(name+".json"))
        rows=point["rows"]
        assert point["model"]==name and point["sha256"]==digest(ROOT/m["weights"])==m["sha256"]
        assert len(rows)==300 and {r["map"]:r["split"] for r in rows}==reference
        assert all(r["steps"]>0 and r["steps"]<=8*len(levels[r["map"]].grid)*len(levels[r["map"]].grid[0]) for r in rows)
        outcomes=Counter(r["outcome"] for r in rows)
        assert set(outcomes)<=set(("EMPTY","BROKEN","LIMIT","SUCCESS"))
        wins={s:sum(r["outcome"]=="SUCCESS" for r in rows if r["split"]==s) for s in SPLITS}
        assert wins==point["wins"]
        c=m["config"]
        scores.append(dict(model=name,family=c["family"],kind=m["kind"],frozen=m["frozen"],variant=c["variant"],seed=c["seed"],
                           **wins,total=outcomes["SUCCESS"],empty=outcomes["EMPTY"],broken=outcomes["BROKEN"],limit=outcomes["LIMIT"],
                           serviced=sum(bool(r["refuels"] or r["repairs"]) for r in rows),
                           both_services=sum(bool(r["refuels"] and r["repairs"]) for r in rows)))
    scores.sort(key=lambda r:(-r["total"],r["model"]))
    with (OUT/"scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=scores[0])
        writer.writeheader()
        writer.writerows(scores)
    by_name={r["model"]:r for r in scores}
    def values(row):
        return [row[k] for k in (*SPLITS,"total")]
    selected=[]
    averages=[]
    replays={}
    paths={}
    verify_names=set()
    for family,label in NAMES.items():
        name=frozen[family]["id"]
        selected.extend(([label,"Основная",*values(by_name[name])],[label,"Дообученная",*values(by_name[name+"-diag"])]))
        verify_names.update((name,name+"-diag"))
        group=[r for r in scores if r["family"]==family and r["kind"]=="primary"]
        assert len(group)==50
        hi=group[0]["total"]
        best=[r for r in group if r["total"]==hi]
        verify_names.add(best[0]["model"])
        averages.append([label,*[f'{mean(r[s] for r in group):.2f}' for s in (*SPLITS,"total")],
                         f'{min(r["total"] for r in group)}–{hi}',len(best)])
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True
    perception.SIDE=spec["input_canvas"]
    for name in sorted(verify_names | {r["model"] for r in scores if r["total"]}):
        m=models[name]
        replays[name],successful=repeat(ROOT/m["weights"],m["config"],read(OUT/"models"/(name+".json"))["rows"],levels,name in verify_names)
        if successful:
            paths[name]=successful
    assert sum(len(v) for v in paths.values())==sum(r["total"] for r in scores)
    save(OUT/"successful-paths.json",paths)
    reserve={}
    with (ROOT/"results/E-0010/reserve.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            if r["split"] in SPLITS:
                assert r["map"] not in reserve and r["split"]==reference[r["map"]]
                reserve[r["map"]]=r["status"]
    assert set(reserve)==set(reference)
    base=[sum(reserve[m]=="SUCCESS" and reference[m]==s for m in reference) for s in SPLITS]
    opt=max(r["total"] for r in scores)
    winners=[r for r in scores if r["total"]==opt]
    proof=dict(models=len(scores),episodes=len(scores)*len(reference),verified_maps=len(proofs),
               independent_model_replays=replays,required_both=dict(require_both),reserve=base,
               winning_episodes=sum(r["total"] for r in scores),zero_models=sum(r["total"]==0 for r in scores),
               best=[r["model"] for r in winners],max_success=opt)
    save(OUT/"verification.json",proof)
    headings=["Семейство","Веса","I / 100","II / 100","III / 100","Всего / 300"]
    positive=[[r["model"],*values(r)] for r in scores if r["total"]]
    notes=[f"Исходные карты и запасы совпали с архивными отпечатками трёх сотен E-0010. "
           f"Для каждой из 300 карт сохранён физически исполненный путь до финиша. Проведены {proof['episodes']:,} поездок "
           "на CUDA с замороженными весами. Каждый эпизод начинался с исходными ресурсами и чистой памятью клеток.",
           f"Совпадения по рисунку стен с учебным корпусом E-0012 — {len(spec['train_overlap'])}, "
           f"с учебным банком диагностических примеров — {len(spec['diagnostic_train_overlap'])}.",
           "## Модели, выбранные до проверки на сотнях",
           table(headings,selected),"## Все 200 основных моделей",
           "Среднее число финишей по 50 моделям семейства. В последних двух столбцах приведены диапазон суммы и число моделей с максимальной суммой в этом семействе.",
           table(["Семейство","I","II","III","Среднее из 300","Диапазон из 300","Достигли верхней границы"],averages),
           "## Все модели, достигшие финиша хотя бы раз",
           f"Максимальная сумма — {opt} из 300. Число моделей с этой суммой — {len(winners)}. Ноль финишей у {proof['zero_models']} моделей. "
           "Эта подборка сделана после просмотра результата на сотнях.",
           table(["Модель","I","II","III","Всего"],positive),
           f"У {sum(require_both.values())} из 300 карт оба начальных запаса меньше длины кратчайшего физического пути до финиша. "
           "На каждой из этих карт для финиша обязательны хотя бы одна заправка и хотя бы один ремонт. "
           f"До обоих типов сервиса роботы добрались в {sum(r['both_services'] for r in scores):,} поездках из {proof['episodes']:,}.",
           f"Базовый автомат frontier-reserve в сохранённых поездках E-0010 на этих же картах завершил "
           f"{base[0]}, {base[1]} и {base[2]} поездок по сотням, всего {sum(base)} из 300.",
           f"Сверены все покартовые записи. Для {sum(map(len,replays.values()))} поездок исход, число ходов и посещения обоих сервисов "
           "проверены повтором из сохранённых весов с исполнением каждого хода.",
           "[Все 204 результата](scores.csv) · [Покартовые исходы](models/) · [Команды всех успешных поездок моделей](successful-paths.json) · [Маршруты проверки проходимости](witnesses.json) · [Сверка](verification.json)."]
    write_text(OUT/"report.md","# E-0013. Нейронные модели на трёх стресс-сотнях\n\n"+"\n\n".join(notes)+"\n")
    print(proof["episodes"],sum(map(len,replays.values())),base,opt,len(winners),flush=True)


if __name__=="__main__":
    main()
