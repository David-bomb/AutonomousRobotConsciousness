"""Сводка 50 архивных обучений и вклад вариантов в проверку на сотнях."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parents[1]
runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((SOURCES / "E-0012/runs").glob("rainbow-*/run.json"))]
with (SOURCES / "E-0013/scores.csv").open(encoding="utf-8-sig", newline="") as data:
    stress = {r["model"]: r for r in csv.DictReader(data) if r["family"]=="rainbow" and r["kind"]=="primary"}
assert len(stress) == len(runs) == 50 and set(stress) == {r["config"]["id"] for r in runs}

def summarize(part):
    stages = []
    for idx in range(3):
        ends = [Counter(r["training"][idx]["outcomes"]) for r in part]
        counts = sum(ends, Counter())
        stages.append(dict(total=sum(counts.values()), outcomes=counts,
                           median_wins=median(r["SUCCESS"] for r in ends),
                           zero_wins=sum(r["SUCCESS"] == 0 for r in ends)))
    gains = [r["screen"]["wins"] - r["initial"]["wins"] for r in part]
    return dict(count=len(part), train_stages=stages,
                before=sum(r["initial"]["wins"] for r in part),
                after=sum(r["screen"]["wins"] for r in part),
                improved=sum(g>0 for g in gains), unchanged=gains.count(0), decreased=sum(g<0 for g in gains),
                hundreds={k:sum(int(stress[r["config"]["id"]][k]) for r in part)
                          for k in ("total","empty","broken","limit","serviced","both_services")})


result = dict(all_runs=summarize(runs),
              variants={i:summarize([r for r in runs if r["config"]["variant"]==i]) for i in range(1,11)})
(HERE/"family.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
