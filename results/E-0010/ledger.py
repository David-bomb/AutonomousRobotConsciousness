"""Сводка полностью измеренных ветвей и список изменений на каждой карте."""
from collections import Counter
import argparse
import json
from statistics import mean

from bench import HERE, SETS
from compare import compare, read

PARENTS = {"anchor":"reserve", "passage":"reserve", "joint":"passage", "roads":"reserve",
           "excursion":"reserve", "range":"excursion", "range-roads":"range",
           "bridge":"passage", "relay":"range-roads", "corridor":"bridge", "cells":"range-roads",
           "fork-budget":"range-roads", "outer":"range-roads", "outer-service":"outer",
           "coupled":"range-roads", "return":"coupled", "witness":"range-roads", "streets":"range-roads",
           "last-unit":"range-roads", "shared":"range-roads", "consensus":"range-roads",
           "homing":"range-roads", "probe-order":"range-roads", "mission":"range-roads",
           "marked-streets":"streets", "many-streets":"marked-streets","mixed-streets":"many-streets",
           "trunk":"range-roads","trunk-reserve":"trunk","street-trunk":"marked-streets",
           "exact-worlds":"marked-streets","street-consensus":"streets","calibrated":"street-consensus",
           "all-approaches":"marked-streets","insured-approaches":"all-approaches",
           "survey-streets":"streets","scouting-sortie":"streets","terminal-streets":"streets",
           "texture-cells":"cells","adaptive-trunk":"street-trunk","stable-streets":"streets","preview":"marked-streets",
           "symmetric-streets":"many-streets","last-mission":"streets","cycle-gate":"street-trunk",
           "margin-gate":"street-trunk","wall-lines":"street-trunk","proven-approach":"street-trunk",
           "individual-stops":"street-trunk","deadline-mission":"street-trunk","deadline-goal":"streets",
           "paired-witness":"street-trunk","common-vote":"marked-streets","street-trunk-reserve":"street-trunk",
           "compact-trunk":"street-trunk","deduced-roads":"street-trunk","conditioned-streets":"many-streets",
           "room-passages":"street-trunk","weighted-rooms":"room-passages","implied-turn":"street-trunk",
           "renewable-routes":"street-trunk","passage-flow":"street-trunk","mission-order":"street-trunk",
           "remote-station":"street-trunk","mandatory-stop":"remote-station","viability-vote":"street-trunk",
           "independent-exits":"street-trunk","stretch-sortie":"street-trunk","conditional-mission":"street-trunk",
           "window-bounds":"street-trunk","uniform-streets":"marked-streets","translated-paths":"street-trunk",
           "room-reserve":"room-passages","weighted-room-reserve":"weighted-rooms","room-evidence":"street-trunk-reserve",
           "no-probe-streets":"street-trunk-reserve","last-direction":"no-probe-streets",
           "known-landing":"street-trunk-reserve","boundary-landing":"known-landing","sampled-sortie":"street-trunk-reserve",
           "local-corridors":"street-trunk-reserve","mission-order-reserve":"mission-order",
           "first-landing":"boundary-landing","shortcut-reserve":"street-trunk-reserve",
           "after-landing":"first-landing","outward-after-landing":"after-landing",
           "goal-window":"street-trunk-reserve","margin-window":"goal-window",
           "smallest-detour":"street-trunk","gauge-detour":"smallest-detour","leg-survival":"street-trunk-reserve",
           "room-mission":"mission-order-reserve","wall-window":"margin-window","matching-survival":"leg-survival",
           "covered-approach":"street-trunk-reserve","exit-flow":"margin-window","boundary-flow":"exit-flow",
           "corridor-flow":"exit-flow","exit-current":"exit-flow","observed-exit":"covered-approach",
           "mission-exit-flow":"mission-order-reserve","path-particles":"street-trunk-reserve",
           "mission-corridor-flow":"mission-exit-flow","sampled-margin":"corridor-flow",
           "surveyed-room-mission":"room-mission","simple-exit-paths":"exit-current","nearest-station":"mission-order-reserve",
           "mission-simple-paths":"simple-exit-paths","service-exit-paths":"simple-exit-paths",
           "service-contact":"mission-order-reserve","gateway-paths":"simple-exit-paths",
           "approach-heading":"mission-simple-paths","feasible-windows":"simple-exit-paths",
           "joint-edge-paths":"simple-exit-paths","station-simple-paths":"mission-simple-paths",
           "useful-service-paths":"service-exit-paths","compatible-loops":"mission-simple-paths",
           "outpost-exits":"useful-service-paths","weighted-loops":"compatible-loops","dominant-loops":"weighted-loops",
           "station-useful-paths":"station-simple-paths","specific-paths":"station-useful-paths",
           "conditional-exits":"simple-exit-paths","observed-loops":"weighted-loops",
           "protected-turn":"station-useful-paths","spare-turn":"protected-turn",
           "second-window":"specific-paths","open-second-window":"second-window","axis-loops":"dominant-loops",
           "perimeter-exits":"specific-paths","perimeter-approach":"perimeter-exits","measured-detours":"specific-paths",
           "bypass-station":"specific-paths","unanimous-bypass":"bypass-station",
           "ordered-trunk":"specific-paths","finish-trunk":"ordered-trunk",
           "return-paths":"specific-paths","last-return-paths":"return-paths","overlapping-windows":"second-window",
           "return-windows":"second-window","probe-pairs":"last-return-paths","probe-then-direct":"second-window",
           "reference-roads":"return-windows","backup-allowance":"second-window","branch-allowance":"second-window",
           "room-lengths":"return-windows","proved-finish":"second-window","proved-windows":"proved-finish",
           "route-shield":"return-windows","unseen-return":"return-windows","unseen-approach":"unseen-return",
           "standby-turn":"return-windows","ranked-standby":"standby-turn",
           "necessary-return":"unseen-return","necessary-approach":"necessary-return",
           "modeled-split":"return-windows","modeled-trunk":"modeled-split",
           "reference-shells":"return-windows","consistent-shells":"reference-shells","sensed-paths":"return-windows",
           "onward-services":"return-windows","standby-exploration":"ranked-standby","visible-standby":"standby-exploration",
           "connected-excursions":"return-windows","conditional-standby":"standby-exploration","conditional-probe":"conditional-standby",
           "serviced-windows":"return-windows","three-windows":"serviced-windows",
           "window-patterns":"return-windows","passage-patterns":"window-patterns",
           "anchored-directions":"unseen-return","mapped-directions":"anchored-directions",
           "mission-shells":"consistent-shells","standby-shells":"mission-shells",
           "linked-services":"ranked-standby","seen-linked-services":"linked-services",
           "exchanged-roads":"ranked-standby","preserved-exchanges":"exchanged-roads",
           "terrain-trials":"exchanged-roads","terrain-rooms":"terrain-trials",
           "mandatory-proof":"ranked-standby","confirmed-supply":"mandatory-proof",
           "street-exits":"ranked-standby","bounded-street-exits":"street-exits",
           "corner-approaches":"ranked-standby","multiple-corners":"corner-approaches",
           "resource-deadline":"mandatory-proof","directed-deadline":"resource-deadline",
           "gauge-exits":"ranked-standby","fitted-gauge-exits":"gauge-exits",
           "side-step":"ranked-standby","paired-side-step":"side-step","two-side-step":"side-step",
           "available-exits":"ranked-standby","flexible-exits":"available-exits",
           "exact-approaches":"ranked-standby","repeated-approaches":"ranked-standby"}


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-missing",action="store_true",help="добавить парные журналы готовых серий")
    args=parser.parse_args()
    root = read("reserve")
    for split,old in zip(SETS,("known1","known2","fresh","val")):
        archive = lines(HERE.parent/"E-0009"/f"{old}-episodes.jsonl")
        for r in archive:
            current = root[r["map"]]
            assert current["split"] == split
            assert all(current[k]==r["agents"]["reserve"][k] for k in ("status","steps","refuels","repairs"))
    reference = {r["map"]:r["parent_loss"] for r in lines(HERE/"reserve--anchor.jsonl") if "parent_loss" in r}
    for row in lines(HERE.parent/"E-0008/trace.jsonl"):
        old,new = row["critical"],reference[row["map"]]
        assert old["step"]==new["step"] and old["move"]==new["move"] and set(old["viable"])==set(new["viable"])
    data = {"reserve":root}
    for key in PARENTS:
        path = HERE/f"{key}.jsonl"
        if path.exists():
            rows = lines(path)
            if len(rows) == len(root):
                by_name = {row["map"]:row for row in rows}
                assert by_name.keys() == root.keys()
                assert all(row["split"] == root[name]["split"] for name,row in by_name.items())
                data[key] = by_name
    if args.compare_missing:
        for child,parent in PARENTS.items():
            if child in data and parent in data:
                compare(parent, child)
    def fmt_rate(a):
        wins = [r["steps"] for r in a if r["status"]=="SUCCESS"]
        return f"{len(wins)} ({mean(wins):.1f})" if wins else "0"
    doc = ["# E-0010. Парные исходы всех законченных серий","",
           "В скобках указано среднее число ходов среди финишей данного автомата. "
           "Все 336 записей корня совпали с E-0009 по исходу, ходам и обоим счётчикам обслуживания. "
           "У независимой поклеточной проверки E-0008 и поиска по станциям E-0010 совпали "
           "номер первого необратимого хода, команда и все спасающие направления на 90 отказах.","",
           "| Автомат | Родитель | I из 100 | II из 100 | III из 100 | val из 36 |",
           "|---|---|---:|---:|---:|---:|"]
    for key,rows in data.items():
        cells = [fmt_rate([r for r in rows.values() if r["split"]==split]) for split in SETS]
        doc.append("| "+" | ".join([f"[`{key}`]({key}.jsonl)",PARENTS.get(key,"—"),*cells])+" |")
    doc += ["","## По сравнению с непосредственным родителем","",
            "Указаны все приобретённые и потерянные финиши. Изменившаяся поездка имеет хотя бы одну другую физическую команду. "
            "В колонке общих финишей даны количества быстрее / столько же / медленнее у потомка.","",
            "| Потомок | Набор | Изменились поездки | Приобретены | Потеряны | Общие финиши |",
            "|---|---|---:|---|---|---|"]
    for child,parent in PARENTS.items():
        if child not in data or parent not in data:
            continue
        detail = HERE/f"{parent}--{child}.jsonl"
        link = f"[`{child}`]({detail.name})" if detail.exists() and len(lines(detail))==336 else child
        for split in SETS:
            a,b = data[parent],data[child]
            names = sorted(k for k,v in b.items() if v["split"]==split)
            win = lambda r:r["status"]=="SUCCESS"
            gains = [k for k in names if win(b[k]) and not win(a[k])]
            losses = [k for k in names if win(a[k]) and not win(b[k])]
            changed = sum(a[k]["actions"]!=b[k]["actions"] for k in names)
            delta = Counter((b[k]["steps"]>a[k]["steps"])-(b[k]["steps"]<a[k]["steps"])
                            for k in names if win(a[k]) and win(b[k]))
            doc.append(f"| {link} | {split} | {changed} | {', '.join(gains) or '—'} | "
                       f"{', '.join(losses) or '—'} | {delta[-1]} / {delta[0]} / {delta[1]} |")
    doc += ["","## Исход и число ходов на каждой карте","",
            "Ф означает финиш, Т — исчерпано топливо, И — исчерпана исправность, Л — лимит ходов.","",
            "| Набор | Карта | "+" | ".join(data)+" |","|---|---|"+"---:|"*len(data)]
    labels = dict(SUCCESS="Ф", EMPTY="Т", BROKEN="И", LIMIT="Л")
    for split in SETS:
        for key in sorted(k for k,v in root.items() if v["split"]==split):
            results = [f"{labels[rows[key]['status']]} {rows[key]['steps']}" for rows in data.values()]
            doc.append("| "+" | ".join([split,key,*results])+" |")
    (HERE/"by-map.md").write_text("\n".join(doc)+"\n",encoding="utf8")
    print(", ".join(data))


if __name__=="__main__":
    main()
