"""Характеристика приобретений и регрессий пары по сложности, геометрии и запасам."""
import argparse
from collections import defaultdict
import json
from statistics import mean

from bench import HERE,SETS,cases
from compare import read
from maze.environment import Environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent")
    p.add_argument("child")
    args=p.parse_args()
    parent,child=read(args.parent),read(args.child)
    stem=f"{args.parent}--{args.child}"
    pairs={r["map"]:r for b in (HERE/f"{stem}.jsonl").read_bytes().splitlines() if (r:=json.loads(b))}
    assert len(pairs)==336
    groups=defaultdict(list)
    for split,name,tier,family,level in cases():
        groups["Набор",split].append(name)
        if split=="val":
            groups["Сложность val",tier].append(name)
            continue
        groups["Сложность стресс-карт",tier].append(name)
        groups["Геометрия стресс-карт",family].append(name)
        view=Environment(level).view
        slack=min(view.fuel-view.fuel_distance,view.health-view.repair_distance)
        label="меньше 2" if slack<2 else ("2 или 3" if slack<4 else "от 4")
        groups["Стартовый запас сверх длины до своего сервиса",label].append(name)
    out=[f"# {args.parent} и {args.child}","","Один и тот же уровень входит в каждую соответствующую группу. "
         "Две лишние единицы позволяют оплатить пробу в неверном направлении и возврат. "
         "Снижение числа ходов на общих финишах означает более быструю поездку потомка.",""]
    for section in dict.fromkeys(k for k,_ in groups):
        out.extend([f"## {section}","","| Группа | Карт | Финиши родителя | Финиши потомка | Приобретено | Потеряно | "
                    "Средняя разница ходов на общих финишах |","|---|---:|---:|---:|---:|---:|---:|"])
        for (kind,label),names in groups.items():
            if section!=kind:
                continue
            a={n for n in names if parent[n]["status"]=="SUCCESS"}
            b={n for n in names if child[n]["status"]=="SUCCESS"}
            delta=[child[n]["steps"]-parent[n]["steps"] for n in a&b]
            out.append(f"| {label} | {len(names)} | {len(a)} | {len(b)} | {len(b-a)} | {len(a-b)} | "
                       + (f"{mean(delta):+.1f}" if delta else "—")+" |")
        out.append("")
    out.extend(["## Все изменения исхода","",
                "Команды указаны в первом расхождении двух траекторий. Последний столбец показывает первый "
                "необратимый ход проигравшего автомата. Нумерация начинается с единицы.","",
                "| Набор | Карта | Финиш у потомка | Ходы родителя / потомка | Первое расхождение | "
                "Команды родителя / потомка | Необратимый ход |","|---|---|---|---:|---:|---|---:|"])
    for split in SETS:
        for n in sorted(k for k,r in pairs.items() if r["split"]==split):
            r=pairs[n]
            if r["verdict"] not in ("приобретён","потерян"):
                continue
            f=r["first"]
            loss=r["parent_loss"] if r["verdict"]=="приобретён" else r["child_loss"]
            out.append(f"| {split} | {n} | {r['verdict']} | {r['parent'][1]} / {r['child'][1]} | "
                       f"{f['step']} | {f['parent']} / {f['child']} | {loss['step']} |")
    out.extend(["",f"[Все 336 парных записей]({stem}.jsonl)"])
    with (HERE/f"profile-{stem}.md").open("x",encoding="utf8") as f:
        f.write("\n".join(out)+"\n")


if __name__=="__main__":
    main()
