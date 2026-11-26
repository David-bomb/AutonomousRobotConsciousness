"""Зафиксировать выбор на разработке или свести уже завершённые парные прогоны."""
import argparse
from collections import Counter
import csv
import json
from statistics import median

from run import ARC, BASE, HERE, PACK, SEED, VARIANTS, digest, maps_from


def rows(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, data):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def summarize(path, names, total, unsolved=()):
    data = rows(path)
    maps = {r["map"] for r in data}
    pairs = {(r["map"],r["variant"]):r for r in data}
    assert len(maps) == total
    assert len(data) == len(pairs) == len(names)*total
    assert set(pairs) == {(m,n) for m in maps for n in names}
    result = {}
    for name in names:
        wins, losses, rescued, common, common_base = [], [], [], [], []
        tiers = dict.fromkeys(sorted({x["tier"] for x in data}),0)
        statuses = Counter()
        for m in sorted(maps):
            a,b = pairs[m,name],pairs[m,BASE]
            assert a["invalid"] == "0" and (a["tier"],a["family"]) == (b["tier"],b["family"])
            statuses[a["status"]] += 1
            if a["status"] == "SUCCESS":
                tiers[a["tier"]] += 1
                if m in unsolved:
                    rescued.append(m)
                if b["status"] == "SUCCESS":
                    common.append(int(a["steps"]))
                    common_base.append(int(b["steps"]))
                else:
                    wins.append(m)
            elif b["status"] == "SUCCESS":
                losses.append(m)
        result[name] = dict(success=statuses["SUCCESS"],tiers=tiers,outcomes=dict(statuses),
                            wins=wins,losses=losses,rescued=rescued,common=len(common),
                            median_steps=median(common),median_base=median(common_base),
                            median_paired_delta=median(x-y for x,y in zip(common,common_base)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("freeze","report"))
    cmd = p.parse_args().command
    old = rows(ARC/"replay/stress/episodes.csv")
    unsolved = {r["map"] for r in old} - {r["map"] for r in old if r["status"]=="SUCCESS"}
    assert len(unsolved)==36
    dev = HERE/"development.csv"
    field_names = ("map","status","steps","repairs","refuels","invalid")
    def baseline(data,name):
        return sorted(tuple(r[k] for k in field_names) for r in data if r["variant"]==name)
    assert baseline(rows(dev),BASE) == baseline(old,"frontier-reserve")
    summary = summarize(dev,VARIANTS,100,unsolved)
    def rank(name):
        r = summary[name]
        return -r["success"],-min(r["tiers"].values()),r["median_paired_delta"],len(name),name
    winner = min((n for n in VARIANTS if n != BASE),key=rank)
    snapshot = dict(variant=winner,controller=digest(HERE/"controller.py"),runner=digest(HERE/"run.py"),
                    scorer=digest(HERE/"score.py"),plan=digest(HERE/"protocol.md"),
                    development=digest(dev),maps=digest(HERE/"fresh.jsonl"),seed=SEED)
    if cmd == "freeze":
        fresh,old_maps = maps_from(HERE/"fresh.jsonl"),maps_from(PACK/"maps.jsonl")
        assert not {r["geometry"] for r in fresh} & {r["geometry"] for r in old_maps}
        assert all(sorted(Counter(r[k] for r in fresh).values())==[25]*4 for k in ("tier","family"))
        save(HERE/"frozen.json",snapshot)
        print(winner,summary[winner])
        return
    assert json.loads((HERE/"frozen.json").read_text(encoding="utf-8")) == snapshot
    fresh = summarize(HERE/"confirmation.csv",[BASE,winner],100)
    val = summarize(HERE/"validation.csv",[BASE,winner],36)
    assert set(r["map"] for r in rows(HERE/"confirmation.csv")) == {r["name"] for r in maps_from(HERE/"fresh.jsonl")}
    union = sorted(set().union(*(set(s["rescued"]) for s in summary.values())))
    report = dict(development=summary,rescued_by_any=union,confirmation=fresh,validation=val)
    save(HERE/"comparison.json",report)
    for k in ("confirmation","validation"):
        print(k,report[k][BASE]["success"],report[k][winner])


if __name__ == "__main__":
    main()
