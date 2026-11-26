"""Что выбирают доступные роботу признаки среди первых достижимых новых окон."""
import json
from collections import Counter

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment
from maze.first_window import window_arrivals
from maze.frontier_reserve import _lengths, _optimistic
from maze.sight_automaton import window_unknown


def main():
    source={r["map"]:r for line in (HERE/"next-observation.jsonl").read_bytes().splitlines() if (r:=json.loads(line))}
    assert len(source)==24
    history=read("street-trunk")
    results=[]
    for split,name,_,_,level in cases():
        if name not in source:
            continue
        event=source[name]
        env=Environment(level)
        for char in history[name]["actions"][:event["step"]-1]:
            env.step(MOVES[int(char)])
        v=env.view
        # Самостоятельный поиск контроллера получает только Observation. Здесь
        # сравниваем его предложения с отдельным перечислением из диагностики.
        offered={(a.point,a.length,a.fuel,a.health) for a in window_arrivals(v) if a.length<=env.limit-env.steps}
        goal=v.position[0]+v.dx,v.position[1]+v.dy
        remaining=_lengths(_optimistic(v.memory),goal)
        scores={}
        for entry in event["exits"]:
            p=tuple(entry["point"])
            new_cells=len(window_unknown(v.memory,p))
            lower=remaining[p]
            for a in entry["arrivals"]:
                least=min(a["fuel"],a["health"])
                length=a["length"]
                # Сначала оценка по указанному правилу, затем длина подъезда.
                scores[p,length,a["fuel"],a["health"]]=dict(
                    viable=a["viable"],near=(-length,),goal=(-lower,-length),stock=(least,-length),
                    margin=(least-lower,-length),new=(new_cells,-length),new_per_step=(new_cells/length,-length))
        assert set(scores)==offered, (name, set(scores)^offered)
        predictions={}
        for rule in ("near","goal","stock","margin","new","new_per_step"):
            best=max(row[rule] for row in scores.values())
            tied=sorted((k for k,row in scores.items() if row[rule]==best),key=lambda k:(k[0],k[1],-k[2],-k[3]))
            predictions[rule]=dict(choice=tied[0],ties=len(tied),safe_ties=sum(scores[k]["viable"] for k in tied),
                                   chosen_safe=scores[tied[0]]["viable"])
        results.append(dict(map=name,split=split,step=event["step"],predictions=predictions))
    (HERE/"window-ranking.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results),encoding="utf8")
    totals=Counter(rule for r in results for rule,p in r["predictions"].items() if p["chosen_safe"])
    print("Первые новые окна",len(results),dict(totals),flush=True)


if __name__=="__main__":
    main()
