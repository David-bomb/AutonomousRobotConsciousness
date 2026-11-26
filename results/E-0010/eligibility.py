"""Какая часть выбора направления действительно доступна для изменения при первом отказе."""
import argparse
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.frontier_reserve import ReserveFrontier
from maze.resource_automaton import known_paths, possible_directions,service_order


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args=parser.parse_args()
    pairs=[json.loads(l) for l in (HERE/"reserve--anchor.jsonl").read_bytes().splitlines()]
    losses={r["map"]:r["parent_loss"] for r in pairs if "parent_loss" in r}
    traces=read("reserve")
    counts=Counter()
    with (HERE/args.output).open("x",encoding="utf8") as output:
        for split,name,tier,family,level in cases():
            event=losses.get(name)
            if event is None:
                continue
            bot,env=ReserveFrontier(),Environment(level)
            actions=traces[name]["actions"]
            for i in range(event["step"]-1):
                move=bot.choose(env.view)
                assert move==MOVES[int(actions[i])]
                env.step(move)
            v=env.view
            active=bot._fallback.mode
            returning=bot._fallback.returning
            service=service_order(v,active,bot.RESERVE)
            previous=(bot._fallback.last_lengths.get(service.mode) if service and v.moved else None)
            move=bot.choose(v)
            assert move.name==event["move"]
            options=possible_directions(v,service,previous) if service else {}
            exact=bool(service and any(known_paths(v.memory,v.position).get(p,(None,))[0]==service.distance
                                       for p in service.sites))
            possible=set(m.name for m in options)
            feasible=set(event["viable"])
            if service is None or bot.activity not in ("На заправку","В мастерскую"):
                reason="другой режим"
            elif not feasible & possible:
                reason="все подходы к выбранному сервису теряют финиш"
            elif returning:
                reason="после обратного хода"
            elif previous is not None and previous<service.distance:
                reason="возрос замер"
            elif exact:
                reason="известен кратчайший путь"
            elif len(possible)<2:
                reason="меньше двух подходов"
            elif move.name not in possible:
                reason="команда вне подходов"
            else:
                reason="выбор среди подходов"
            counts[reason]+=1
            r=dict(map=name,split=split,tier=tier,family=family,reason=reason,**event,
                   previous=previous,returning=returning,mode=bot.activity,
                   target=service.mode.name if service else None,options=sorted(possible),exact=exact)
            output.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(counts)


if __name__=="__main__":
    main()
