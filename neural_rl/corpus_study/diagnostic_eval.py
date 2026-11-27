"""Исполнение замороженных контроллеров на задачах и в заданных диагностических позициях."""
from collections import Counter,defaultdict
from dataclasses import replace
import gzip
import json
import torch

from maze.datasets import Case
from maze.environment import Environment
from maze.interface import Move
from maze.levels import Level
from neural_rl.experiment import digest,read,save
from neural_rl.networks import batch
from neural_rl.perception import encode
from .corpus import MOVES,edges,tree
from .diagnostics import FOLDER,NAMES,probe_view
from .learn import DirectNetwork,evaluate_rows,setup
from .report import NAMES as FAMILY_NAMES,table,write_text
from .session import OUT,check_time,log


class DiagnosticPool:
    def __init__(self,records):
        self.cases=[Case(r["name"],r["group"],Level.parse(r["source"])) for r in records]

    def start(self,case):
        return Environment(case.level)


def expected_moves(row,view):
    """Эталон сравнения для группы с единственным нужным ресурсом или известным обходом."""
    if row["group"] not in (4,6,11):
        return None
    level=Level.parse(row["source"])
    floor=tuple("".join("#" if c=="#" else "." for c in line) for line in level.grid)
    graph=edges(floor)
    goal_dist=tree(graph,level.goal)[1][view.position]
    targets=[level.goal]
    needed="F" if view.fuel<goal_dist else "R" if view.health<goal_dist else None
    if needed:
        targets=[(x,y) for y,line in enumerate(level.grid) for x,c in enumerate(line) if c==needed]
    distances=[tree(graph,q)[1] for q in targets]
    def remaining(p): return min(d.get(p,10**9) for d in distances)
    x,y=view.position
    return [MOVES.index(Move((q[0]-x,q[1]-y))) for q in graph[view.position]
            if remaining(q)==remaining(view.position)-1]


def main():
    frozen=read(OUT/"frozen.json")
    with gzip.open(FOLDER/"control.jsonl.gz","rt",encoding="utf-8") as stream:
        records=[json.loads(l) for l in stream]
    if len(records)!=1000: raise RuntimeError("Неполный диагностический контроль")
    pool=DiagnosticPool(records)
    by_group=defaultdict(dict)
    pair_rows=[]
    group_rows=[]
    for family,choice in frozen.items():
        check_time()
        folder=OUT/"runs"/choice["id"]
        config=read(folder/"run.json")["config"]
        if digest(folder/"weights.pt")!=choice["sha256"]: raise ValueError("Вес модели изменился")
        setup(config["seed"])
        net=DirectNetwork(config).cuda()
        net.load_state_dict(torch.load(folder/"weights.pt",weights_only=True,map_location="cuda"))
        outcomes=evaluate_rows(net,pool,chunk=48)
        lookup={r["map"]:r for r in outcomes}
        groups=defaultdict(list)
        with torch.no_grad():
            for start in range(0,len(records),48):
                block=records[start:start+48]
                views=[probe_view(r) for r in block]
                actions=net(batch([encode(v,False) for v in views]))[0].argmax(1).tolist()
                for r,v,a in zip(block,views,actions):
                    expected=expected_moves(r,v)
                    groups[r["group"]].append(dict(**lookup[r["name"]],pair=r["pair"],choice=a,expected=expected))
        report={}
        for group,rows in groups.items():
            counts=Counter(r["outcome"] for r in rows)
            wins=counts["SUCCESS"]
            by_group[group][family]=wins
            report[str(group)]=dict(wins=wins,outcomes=dict(counts),rows=rows)
            if group in (4,5,6,11,19):
                changed=correct=0
                for j in range(0,50,2):
                    a,b=rows[j:j+2]
                    if group==19:
                        dx,dy=MOVES[a["choice"]].value
                        changed+=MOVES[b["choice"]]!=Move((-dy,dx))
                    else:
                        changed+=a["choice"]!=b["choice"]
                    for r in (a,b):
                        if r["expected"] is not None: correct+=r["choice"] in r["expected"]
                if group==11 and changed: raise RuntimeError("Разные ходы при одинаковом входе")
                pair_rows.append([FAMILY_NAMES[family],group,changed,"—" if group not in (4,6,11) else f"{correct}/50"])
        save(FOLDER/f"score-{family}.json",dict(id=choice["id"],hash=choice["sha256"],groups=report))
        log("diagnostic_control_completed",id=choice["id"],scenarios=len(records),
            wins=sum(r["outcome"]=="SUCCESS" for r in outcomes))
    for group in range(1,21):
        group_rows.append([group,NAMES[group-1],*[by_group[group][f] for f in FAMILY_NAMES]])
    write_text(OUT/"diagnostic-results.md","## Двадцать диагностических групп\n\nФинишей из 50 в каждой группе. "
               "Сценарии контрольного банка физически подтверждены и отделены от независимых учебных примеров.\n\n"+
               table(["Группа","Ситуация",*FAMILY_NAMES.values()],group_rows)+
               "\n\nВ группах 4–6 и 11 посчитаны пары с разным выбранным ходом. В группе 19 — несовпадения после поворота назад. "
               "Эталонный первый ход для групп 4, 6 и 11 проверяется в оценщике по полной карте после выдачи решения сетью.\n\n"+
               table(["Семейство","Группа","Различий из 25 пар","Эталонный ход"],pair_rows)+"\n")


if __name__=="__main__": main()
