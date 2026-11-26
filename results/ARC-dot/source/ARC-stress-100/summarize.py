"""Сводки без выбора победителя по данным настройки."""
from collections import Counter
import csv
from pathlib import Path
from statistics import median
import sys

source, out = map(Path,sys.argv[1:])
rows=list(csv.DictReader(source.open(encoding="utf-8")))
agents=sorted({r['variant'] for r in rows})
targets=('H1','H2','H3','H4')
families=('corridors','branching','loops','rooms')
assert len(rows)==100*len(agents) and len({(r['variant'],r['map']) for r in rows})==len(rows)
assert len({r['map'] for r in rows}) == 100
with out.joinpath("summary.csv").open("w",newline='',encoding="utf-8") as f, out.joinpath("by-family.csv").open("w",newline='',encoding="utf-8") as ff:
    w=csv.writer(f); wf=csv.writer(ff)
    w.writerow(("variant",*targets,"success","fuel_empty","broken","limit","median_success_steps",
                "median_success_stretch","invalid_moves"))
    wf.writerow(("variant",*families))
    for k in sorted(agents,key=lambda k:(-sum(r["status"]=="SUCCESS" and r['variant']==k for r in rows),k)):
        block=[r for r in rows if r['variant']==k]
        outcomes=Counter(r['status'] for r in block)
        wins=[r for r in block if r['status']=='SUCCESS']
        d=[sum(r["tier"]==t for r in wins) for t in targets]
        assert all(int(r['steps'])>=int(r['optimum']) for r in wins)
        m=median(int(r['steps']) for r in wins) if wins else ""
        stretch=round(median(int(r['steps'])/int(r['optimum']) for r in wins),4) if wins else ""
        w.writerow((k,*d,*(outcomes[s] for s in ('SUCCESS','EMPTY','BROKEN','LIMIT')),m,stretch,
                    sum(int(r['invalid']) for r in block)))
        wf.writerow((k,*(sum(r["family"]==family for r in wins) for family in families)))
        print(k,*d,*(outcomes[s] for s in ('SUCCESS','EMPTY','BROKEN','LIMIT')),m,stretch)
