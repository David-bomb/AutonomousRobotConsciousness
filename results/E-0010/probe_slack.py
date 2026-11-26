"""Когда перед ошибкой наведения исчез запас на один выход с возвратом."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment


def main():
    audited={r["map"]:r for b in (HERE/"target-or-direction.jsonl").read_bytes().splitlines()
             if (r:=json.loads(b))["viable_downhill"]}
    runs=read("street-trunk")
    stats=Counter()
    with (HERE/"probe-slack.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in audited:
                continue
            env=Environment(level)
            chosen=audited[name]["selected"]
            event=audited[name]["critical"]
            def readings():
                if chosen=="F":
                    return env.fuel,env.view.fuel_distance,env.refuels,env.repairs
                return env.health,env.view.repair_distance,env.repairs,env.refuels
            readings_now=readings()
            anchor=dict(step=0,stock=readings_now[0],d=readings_now[1],other=readings_now[3])
            crossed=None
            spent=Counter()
            for c in runs[name]["actions"][:event["step"]-1]:
                before=readings_now
                env.step(MOVES[int(c)])
                readings_now=readings()
                stock,d,visits,other=readings_now
                if visits>before[2]:
                    anchor=dict(step=env.steps,stock=stock,d=d,other=other)
                    crossed=None
                    spent.clear()
                    continue
                delta=d-before[1]
                spent[delta]+=1
                if before[0]-before[1]>=2 and stock-d<2:
                    crossed=dict(step=env.steps,move=MOVES[int(c)].name,
                                 stock_before=before[0],d_before=before[1],stock=stock,d=d)
            stock,d,_,other=readings_now
            start_slack=anchor["stock"]-anchor["d"]
            assert start_slack-spent[0]-2*spent[1]==stock-d
            stats[start_slack<2,stock-d<2]+=1
            out.write(json.dumps(dict(map=name,split=split,critical=event,selected=chosen,anchor=anchor,
                                      start_slack=start_slack,remaining_slack=stock-d,first_below_two=crossed,
                                      intervening_other_visits=other-anchor["other"],gauge_changes=spent))+"\n")
    print(stats)


if __name__=="__main__":
    main()
