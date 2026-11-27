"""Recompute lab-director dev scores from complete episode logs."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEV=HERE.parents[3]/"University/Diplom/results/E-0016/data/dev.jsonl"
TIERS=("H1","H2","H3","H4")
RUNS=[
    ("frontier-search","Строгая разведка","packed-dev.json",None),
    ("two-readings-search","Два датчика и признаки направления","dev.physical.json","ablated.physical.json"),
    ("transferred-frontier","Перенос разведки на super1500","dev.physical.json","ablated.physical.json"),
    ("return-search","Продолжения после двух сервисов","dev.physical.json","ablated.physical.json"),
    ("renewable-search","Возобновляемое обслуживание","dev.physical.json","ablated.physical.json"),
    ("proper-search","Выбор кратчайшего пути к новому наблюдению","dev.json","ablated.json"),
    ("geodesic-search","Первое неизвестное ребро","dev.json","ablated.json"),
    ("service-window/fixed","Переход по полному окну сервисов","dev.json","ablated.json"),
    ("service-window/learned","Обучаемый переход по окнам сервисов","dev.json","ablated.json"),
    ("handover-search","Терминальная настройка ранних оценок","dev.json","ablated.json"),
    ("service-votes","Числовая оценка дорог до сервиса","dev.json","ablated.json"),
    ("final-leg","Терминальная настройка после обслуживания","dev.json","ablated.json"),
    ("layout-search","Выбор по расположению станций","dev.json","ablated.json"),
    ("symmetry-search","Оценка раннего хода в восьми системах координат","dev.json","ablated.json"),
    ("origin-search","Совместимость пути с открытой областью и запасами","dev.json","ablated.json"),
    ("tree-search","Условная модель дорог без циклов","dev.json","ablated.json"),
    ("latent-finish","Оценка продолжения после безопасной разведки","dev.json","ablated.json"),
    ("panel-search","Совместный прогноз пространственных и ранних сетей","dev.json","ablated.json"),
    ("rollout-search","Обучение на полных продолжениях после одного хода","dev.json","ablated.json"),
    ("protected-search/fixed","Две ранние поправки с фиксированным приоритетом","dev.json","ablated.json"),
    ("protected-search/learned","Поправки с условной моделью помещений и петель","dev.json","ablated.json"),
    ("lookahead-search","Прогноз двух физических ходов с будущим наблюдением","dev.json","ablated.json"),
    ("exact-search","Совместимость показаний и маршрута обслуживания","dev.json","ablated.json"),
    ("departure-search","Выбор выхода из изученной части карты","dev.json","ablated.json"),
    ("paired-search","Попарное обучение на последствиях альтернативных ходов","dev.json","ablated.json"),
    ("horizon-search","Выбор раннего хода по прогнозу на 3–7 шагов","dev.json","ablated.json"),
    ("boundary-search","Выбор границы изученной области по путям от удалённых целей","dev.json","ablated.json"),
    ("combination-search","Перенос ранних и поздних решений на две новые сотни","dev.json","ablated.json")]


def read(p):
    return json.loads(p.read_bytes())


def main():
    dev_bytes=DEV.read_bytes()
    dev_hash=sha256(dev_bytes).hexdigest()
    cases={r["name"]:r for r in map(json.loads,dev_bytes.splitlines())}
    assert Counter(r["tier"] for r in cases.values())==Counter(dict.fromkeys(TIERS,25))
    lines=["# Полные поездки моделей управляющей лаборатории","",
           "Все результаты относятся к одной новой валидационной сотне E-0016. У каждой сложности 25 карт. "
           "В основной серии робот действует по текущему наблюдению. В отдельной серии он дополнительно получает два предыдущих реальных показания. "
           "В парной проверке с отключением оба прежних значения пусты на каждом шаге.","",
           "| Модель | Вход | Финиши / 100 | H1 | H2 | H3 | H4 | Оба сервиса | Лимит | Всего ходов | Данные |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for folder,title,physical,absence in RUNS:
        base=HERE/folder
        for tag,rel in (("Два показания" if absence else "Текущий снимок",physical),("Показания отключены",absence)):
            if rel is None or not (base/rel).is_file():continue
            obj=read(base/rel)
            assert obj.get("dev_sha256",dev_hash)==dev_hash
            episodes=obj["episodes"]
            assert len(episodes)==100 and {r["map"] for r in episodes}==set(cases)
            tiers=Counter();counts=Counter()
            for e in episodes:
                ref=cases[e["map"]]
                assert e["tier"]==ref["tier"] and e.get("geometry",ref["geometry"])==ref["geometry"]
                assert e["outcome"] in {"SUCCESS","BROKEN","EMPTY","LIMIT"} and e["steps"]>0
                tiers[e["tier"]]+=e["outcome"]=="SUCCESS"
                counts["limit"]+=e["outcome"]=="LIMIT"
                counts["both"]+=bool(e["refuels"] and e["repairs"])
                counts["steps"]+=e["steps"]
            wins=sum(tiers.values())
            if "wins" in obj:assert obj["wins"]==[wins]
            if "summary" in obj:assert obj["summary"]["wins"]==wins
            signature=obj.get("model_sha256") or obj.get("weights_sha256")
            if signature:assert sha256((base/"model.pt").read_bytes()).hexdigest()==signature
            elif (base/"frozen.json").exists():
                frozen=read(base/"frozen.json")
                assert obj["coefficients"]==[frozen["coefficients"]]
            fields=[title,tag,wins,*[tiers[k] for k in TIERS],counts["both"],counts["limit"],counts["steps"],
                    f"[Журнал]({folder}/{rel})"]
            lines.append("| "+" | ".join(map(str,fields))+" |")
    lines.extend(["",f"SHA-256 валидационного набора · {dev_hash}",""])
    path=HERE/"lab-measurements.md"
    text="\n".join(lines)
    if not path.exists() or path.read_text(encoding="utf8")!=text:
        tmp=path.with_suffix(".tmp")
        tmp.write_text(text,encoding="utf8");tmp.replace(path)
    print(path)


if __name__=="__main__":main()
