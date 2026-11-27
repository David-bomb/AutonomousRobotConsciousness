"""Парные контрасты на одних геометриях с сохранением пар внутри зерна."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")
CONTRASTS = {"R_at_C0": (-1,1,0,0), "R_at_C1": (0,0,-1,1),
             "C_at_R0": (-1,0,1,0), "C_at_R1": (0,-1,0,1), "interaction": (1,-1,-1,1)}
MEASURES = ("finish", "fuel_visit", "repair_visit", "both", "return_R0", "return_R1")


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(row):
    return ("F" if row["fuel"] < row["direct"] else "") + ("R" if row["health"] < row["direct"] else "") or "0"


def counts(rows):
    return dict(n=len(rows), finishes=sum(r["finish"] for r in rows),
                visited_fuel=sum(r["fuel_visit"] for r in rows),
                visited_repair=sum(r["repair_visit"] for r in rows),
                visited_both=sum(r["both"] for r in rows),
                outcomes=dict(Counter(r["outcome"] for r in rows)),
                mean_return_R0=float(np.mean([r["return_R0"] for r in rows])),
                mean_return_R1=float(np.mean([r["return_R1"] for r in rows])))


def by_group(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[str(key(r))].append(r)
    return {k:counts(group) for k,group in sorted(groups.items())}


def read_train(record, index):
    out = []
    for h in record["history"]:
        groups = defaultdict(Counter)
        by_size = defaultdict(Counter)
        direct_sum = 0
        for name, starts in h["maps"].items():
            meta = index[name]
            ends = h["finished"].get(name, {})
            tallies = dict(started=starts, completed=sum(ends.values()), wins=ends.get("SUCCESS",0))
            assert 0 <= tallies["wins"] <= tallies["completed"] <= starts
            groups[need(meta)].update(tallies)
            by_size[str(meta["stage"])].update(tallies)
            direct_sum += meta["direct"]*starts
        totals = sum(groups.values(), Counter())
        assert totals["completed"] == sum(h["outcomes"].values())
        assert totals["wins"] == h["outcomes"].get("SUCCESS",0)
        assert 0 <= totals["started"] - totals["completed"] <= record["config"]["lanes"]
        out.append(dict(stage=h["stage"], started=totals["started"], completed=totals["completed"],
                        wins=totals["wins"], by_need=groups, by_size=by_size,
                        direct_sum=direct_sum, unique_scenarios=len(h["maps"]), outcomes=h["outcomes"]))
    return out


def interval(deltas, boot):
    # (зёрна, геометрии). Столбец уже усреднён по четырём поездкам геометрии.
    per_seed = deltas.mean(axis=1)
    estimates = deltas.mean(axis=0)[boot].mean(axis=1)
    cl, ch = np.quantile(estimates, [.025,.975])
    result = dict(mean=float(per_seed.mean()), map_ci95=[float(cl),float(ch)],
                  by_seed=per_seed.tolist())
    if len(per_seed) == 5:
        sd = float(per_seed.std(ddof=1))
        half = 2.776445105 * sd / np.sqrt(5)
        result.update(seed_sd=sd, seed_t95=[float(per_seed.mean()-half), float(per_seed.mean()+half)])
    return result


def summarize(phase, index, reg):
    seeds = [c["seed"] for c in reg["configs"]][:1 if phase == "technical" else 5]
    reports, bank, names = {}, {}, []
    train_stats = {}
    for seed in seeds:
        reports[seed], bank[seed], train_stats[seed] = {}, {}, {}
        for arm in ARMS:
            folder = OUT/phase/str(seed)/arm
            proof = read(folder/"verified.json")
            assert proof["registration"] == digest(OUT/"registration.json")
            assert proof["reference_rows"] == 240 and proof["final_weights_unchanged"]
            assert not proof["cuda_initialized"] and proof["cpu_threads"] <= 2
            for p,h in proof["files"].items():
                assert digest(folder/p) == h
            train = read(folder/"training.json")
            assert train["initial"] == reg["initial"][str(seed)]
            train_stats[seed][arm] = read_train(train, index)
            rows = read(folder/"validation.json")
            for r in rows:
                assert all(r[k] == v for k,v in index[r["map"]].items())
                r.update(finish=int(r["outcome"]=="SUCCESS"), fuel_visit=int(r["refuels"]>0),
                         repair_visit=int(r["repairs"]>0), both=int(min(r["refuels"],r["repairs"])>0))
            if not names:
                names = [r["map"] for r in rows]
            assert [r["map"] for r in rows] == names
            reports[seed][arm] = counts(rows)
            bank[seed][arm] = rows

    geom_list = [index[n]["geometry"] for n in names]
    geometry_groups = np.array([[j for j,k in enumerate(geom_list) if k==g] for g in sorted(set(geom_list))])
    assert geometry_groups.shape == (60,4)
    boot = np.random.default_rng(160016).integers(60, size=(20000,60))
    all_arrays = {m:np.array([[[r[m] for r in bank[s][a]] for a in ARMS] for s in seeds]) for m in MEASURES}
    comparisons, changes = {}, {}
    for label, coefficients in CONTRASTS.items():
        comparisons[label] = {}
        if label != "interaction":
            changes[label] = {}
        for metric, array in all_arrays.items():
            diffs = (array*np.array(coefficients)[None,:,None]).sum(axis=1)
            comparisons[label][metric] = interval(diffs[:,geometry_groups].mean(axis=2), boot)
            if metric == "finish" and label != "interaction":
                changes[label] = {s:{"gained":[names[i] for i in np.flatnonzero(d > 0)],
                                     "lost":[names[i] for i in np.flatnonzero(d < 0)]}
                                  for s,d in zip(seeds,diffs)}
    grouped = {}
    for arm in ARMS:
        rows = [r for seed in seeds for r in bank[seed][arm]]
        grouped[arm] = dict(all=counts(rows), by_stage=by_group(rows, lambda r:r["stage"]),
                            by_deficit=by_group(rows, need),
                            by_stage_and_deficit=by_group(rows, lambda r:f"{r['stage']}/{need(r)}"))
    return dict(seeds=seeds, per_seed=reports, pooled=grouped, contrasts=comparisons, training=train_stats,
                bootstrap=dict(seed=160016,resamples=20000,geometries=60,scenarios_per_geometry=4),
                changes=changes)


def tables(phase, result):
    title = "Технический квартет" if phase == "technical" else "Пять квартетов полного бюджета"
    lines = [f"# {title}", "", "## Полные поездки", "",
             "| Зерно | " + " | ".join(ARMS) + " |", "|---|---:|---:|---:|---:|"]
    for seed, arms in result["per_seed"].items():
        lines.append(f"| {seed} | " + " | ".join(f"{arms[a]['finishes']}/240" for a in ARMS) + " |")
    lines += ["", "По каждой группе указаны число финишей и число поездок. F означает начальный дефицит топлива, "
              "R — исправности, FR — обоих запасов. Уровни по размеру общие для всех ветвей.", "",
              "| Группа | " + " | ".join(ARMS) + " |", "|---|---:|---:|---:|---:|"]
    for kind in ("by_stage","by_deficit"):
        for key in result["pooled"][ARMS[0]][kind]:
            cells = [result["pooled"][a][kind][key] for a in ARMS]
            label = "Ступень" if kind == "by_stage" else "Дефицит"
            lines.append(f"| {label} {key} | " + " | ".join(f"{c['finishes']}/{c['n']}" for c in cells) + " |")
    lines += ["", "Число поездок с заправкой / ремонтом / обоими заездами. В каждой строке общий знаменатель для четырёх ветвей.", "",
              "| Группа | Поездок на ветвь | " + " | ".join(ARMS) + " |", "|---|---:|---:|---:|---:|---:|"]
    for kind, label in (("by_stage","Ступень"), ("by_deficit","Дефицит")):
        for key in result["pooled"][ARMS[0]][kind]:
            cells = [result["pooled"][a][kind][key] for a in ARMS]
            lines.append(f"| {label} {key} | {cells[0]['n']} | " + " | ".join(
                f"{c['visited_fuel']}/{c['visited_repair']}/{c['visited_both']}" for c in cells) + " |")
    lines += ["", "## Какие поездки начались на обучении", "",
              "Первые три числа — начато, завершено и финишировало. Следующие четыре — сколько начатых "
              "поездок имели соответствующий дефицит. Последний столбец — средняя прямая длина пути у начатых поездок.", "",
              "| Ветвь | Ступень | Начато | Завершено | Финиш | 0 | F | R | FR | Длина пути |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for j in range(3):
            stages = [a[arm][j] for a in result["training"].values()]
            sums = [sum(h[k] for h in stages) for k in ("started","completed","wins")]
            mix = [sum(h["by_need"].get(k,{}).get("started",0) for h in stages) for k in ("0","F","R","FR")]
            lines.append("| " + " | ".join(map(str,[arm,j+1,*sums,*mix,
                         f"{sum(h['direct_sum'] for h in stages)/sums[0]:.1f}"])) + " |")
    lines += ["", "## Смена исхода в парных поездках", "",
              "Числа в ячейке — полученные / утраченные финиши второго условия относительно первого. "
              "Имена карт по каждому зерну сохранены в summary.json, поле changes.", "",
              "| Зерно | R1−R0, C0 | R1−R0, C1 | C1−C0, R0 | C1−C0, R1 |",
              "|---|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        cells = [result["changes"][label][seed] for label in result["changes"]]
        lines.append(f"| {seed} | " + " | ".join(f"{len(c['gained'])}/{len(c['lost'])}" for c in cells)+" |")
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("technical","full"))
    phase = parser.parse_args().phase
    reg = read(OUT/"registration.json")
    index = json.loads(gzip.decompress((OUT/"scenarios.json.gz").read_bytes()))
    result = summarize(phase, index, reg)
    save(OUT/phase/"summary.json", result)
    (OUT/phase/"tables.md").write_text(tables(phase, result), encoding="utf8")
    print({k:v["finish"] for k,v in result["contrasts"].items()})
