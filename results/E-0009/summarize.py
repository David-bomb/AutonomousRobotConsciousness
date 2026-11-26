"""Сводка и читаемая таблица изменений по сырым записям опыта."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median

from run import HERE, FILES, rows
from maze.grid import destination, locations, wave
from maze.interface import Move


def compare(maps, name):
    pairs = [(m, m["agents"]["reserve"], m["agents"][name]) for m in maps]
    common = [new["steps"]-base["steps"] for _, base, new in pairs
              if base["status"] == new["status"] == "SUCCESS"]
    gained = [m["map"] for m, base, new in pairs if base["status"] != "SUCCESS" and new["status"] == "SUCCESS"]
    lost = [m["map"] for m, base, new in pairs if base["status"] == "SUCCESS" and new["status"] != "SUCCESS"]
    probes = [p for _, _, new in pairs for p in new["probes"]]
    changed = [(base, new) for _, base, new in pairs if "first" in new]
    shared_failures = [new["critical"]["step"] - base["critical"]["step"] for base, new in changed
                       if base["status"] != "SUCCESS" and new["status"] != "SUCCESS"]
    first_by_result = {}
    for base, new in changed:
        group = status_pair(base, new)
        counts = first_by_result.setdefault(group, Counter())
        p, f = new["probes"][0], new["first"]
        counts["maps"] += 1
        counts["information"] += bool(p["removed"])
        counts["excluded_own"] += p["move"] in p["removed"]
        counts["nearer"] += p["sensor_delta"] == -1
        counts["farther"] += p["sensor_delta"] == 1
        counts["both_viable"] += f["base_viable"] and f["viable"]
        counts["saved_immediately"] += not f["base_viable"] and f["viable"]
        counts["lost_immediately"] += f["base_viable"] and not f["viable"]
        counts["better_service_step"] += new["base_delta"] > p["sensor_delta"]
        counts["worse_service_step"] += new["base_delta"] < p["sensor_delta"]
    return dict(maps=len(maps), reserve=sum(base["status"] == "SUCCESS" for _,base,_ in pairs),
                finishes=len(common)+len(gained), gains=gained, losses=lost, divergences=len(changed),
                first_by_result=first_by_result,
                first_viability=dict(Counter(f'{int(n["first"]["base_viable"])}/{int(n["first"]["viable"])}'
                                             for _, n in changed)),
                shared_success=dict(total=len(common), faster=sum(x<0 for x in common),
                                    equal=common.count(0), slower=sum(x>0 for x in common),
                                    median_delta=median(common) if common else None,
                                    mean_delta=mean(common) if common else None),
                shared_diverged_failures=dict(total=len(shared_failures), later=sum(x>0 for x in shared_failures),
                                              same=shared_failures.count(0), earlier=sum(x<0 for x in shared_failures)),
                probes=len(probes), detours=sum(p["detour"] for p in probes),
                informative=sum(bool(p["removed"]) for p in probes),
                informative_base=sum(bool(p["base_removed"]) for p in probes),
                more=sum(len(p["removed"])>len(p["base_removed"]) for p in probes),
                fewer=sum(len(p["removed"])<len(p["base_removed"]) for p in probes),
                discovered_own_error=sum(p["move"] in p["removed"] for p in probes),
                excluded_own_only=sum(p["removed"] == [p["move"]] for p in probes),
                reading_up=sum(p["sensor_delta"] > 0 for p in probes))


def status_pair(base, new):
    if base["status"] == new["status"] == "SUCCESS":
        return "Оба финишировали"
    if new["status"] == "SUCCESS":
        return "Приобретён финиш"
    if base["status"] == "SUCCESS":
        return "Потерян финиш"
    return "Оба остановились"


def table(maps):
    direction = dict(NORTH="С", EAST="В", SOUTH="Ю", WEST="З", NE="СВ", SE="ЮВ", SW="ЮЗ", NW="СЗ")
    status = dict(SUCCESS="финиш", EMPTY="без топлива", BROKEN="поломка", LIMIT="лимит")
    lines = ["# Все карты с изменившейся траекторией", "",
             "В каждой строке сначала указан закреплённый автомат, затем новый. Столбец «Ход расхождения» показывает первое различное действие из одной клетки. В последнем столбце приведён первый ход потери достижимости у каждой проигравшей стороны. Карту с совпадающими решениями двух новых автоматов показываем один раз. Координаты с нуля, от верхнего левого угла.", "",
             "[Отчёт](../../docs/experiments/E-0009-sight.md) · [Машинная сводка](summary.json)"]
    lines.extend(["", "## Утраченные и приобретённые финиши", "",
                  "| Серия | Агент | Перестал проходить | Впервые прошёл относительно базы |",
                  "|---|---|---|---|"])
    for split, cases in maps.items():
        for agent in ("sight-route", "sight-detour"):
            result = compare(cases, agent)
            lines.append(f'| {split} | {agent} | {", ".join(result["losses"]) or "—"} | '
                         f'{", ".join(result["gains"]) or "—"} |')
    for split, cases in maps.items():
        lines.extend(["", "## " + dict(known1="Первая знакомая сотня", known2="Вторая знакомая сотня",
                                       fresh="Новая сотня", val="Валидационные карты")[split], "",
                      "| Карта | Вариант | Итог сравнения | Ход расхождения, клетка и направление | Исход и ходы, база → вариант | Первый необратимый ход, база / вариант |",
                      "|---|---|---|---|---|---|"])
        for case in cases:
            agents = case["agents"]
            base = agents["reserve"]
            entries = [("route", agents["sight-route"]), ("detour", agents["sight-detour"])]
            if entries[0][1] == entries[1][1]:
                entries = [("оба", entries[0][1])]
            for name, new in entries:
                if "first" not in new:
                    continue
                f = new["first"]
                pos = ", ".join(map(str, f["position"]))
                outcomes = " → ".join(f'{status[a["status"]]} {a["steps"]}' for a in (base,new))
                losses = " / ".join(str(a.get("critical",{}).get("step","—")) for a in (base,new))
                lines.append(f'| {case["map"]} | {name} | {status_pair(base,new)} | {f["step"]}, ({pos}), '
                             f'{direction[f["base"]]} → {direction[f["move"]]} | {outcomes} | {losses} |')
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="папка с четырьмя журналами")
    parser.add_argument("summary", type=Path)
    parser.add_argument("details", type=Path)
    args = parser.parse_args()
    maps = {s: rows(args.directory/f"{s}-episodes.jsonl") for s in FILES}
    for s, cases in maps.items():
        source = rows(FILES[s])
        assert len(cases) == len(source) and [m["map"] for m in cases] == [r["name"] for r in source]
        for case, item in zip(cases, source):
            grid = item["grid"] if s != "val" else item["lines"][2:]
            fields = {}
            for agent in ("sight-route", "sight-detour"):
                new = case["agents"][agent]
                if "first" not in new:
                    continue
                p = new["probes"][0]
                kind = "F" if p["service"] == "FUEL" else "R"
                if kind not in fields:
                    fields[kind] = {q:d for q,d,_ in wave(grid, locations(grid, kind))}
                distances, origin = fields[kind], tuple(p["position"])
                def delta(move):
                    q = destination(grid, origin, Move[move])
                    return distances[q]-distances[origin]
                assert delta(p["move"]) == p["sensor_delta"]
                new["base_delta"] = delta(p["base"])
        if s in ("known1", "known2", "val"):
            filename = {"known1":"development.csv", "known2":"confirmation.csv", "val":"validation.csv"}[s]
            with (HERE.parent/"E-0007"/filename).open(encoding="utf-8", newline="") as f:
                archive = {r["map"]:r for r in csv.DictReader(f) if r["variant"] == "reserve"}
            for m in cases:
                base, old = m["agents"]["reserve"], archive[m["map"]]
                assert base["status"] == old["status"] and all(base[k] == int(old[k]) for k in ("steps","refuels","repairs"))
    summary = {}
    for s, cases in maps.items():
        summary[s] = {}
        for agent in ("sight-route", "sight-detour"):
            summary[s][agent] = compare(cases, agent)
            tiers = sorted({r["tier"] for r in cases})
            summary[s][agent]["by_tier"] = {t: compare([r for r in cases if r["tier"] == t], agent) for t in tiers}
    with args.summary.open("x", encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    with args.details.open("x", encoding="utf-8", newline="\n") as f:
        f.write(table(maps))


if __name__ == "__main__":
    main()
