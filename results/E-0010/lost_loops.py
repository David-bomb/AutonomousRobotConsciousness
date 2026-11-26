"""Как изменится достижимость после удаления петель без новых клеток и обслуживания."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read,transition
from maze.environment import Environment


def shorten(env,actions):
    kept=[]
    chain=[env.position]
    indices={env.position:0}
    start=0
    removed=[]
    for c in actions:
        before=env.view.memory,env.refuels,env.repairs
        kept.append(c)
        env.step(MOVES[int(c)])
        if before!=(env.view.memory,env.refuels,env.repairs):
            chain=[env.position]
            indices={env.position:0}
            start=len(kept)
        elif env.position in indices:
            i=indices[env.position]
            removed.append(len(kept)-start-i)
            del kept[start+i:]
            for q in chain[i+1:]:
                del indices[q]
            del chain[i+1:]
        else:
            indices[env.position]=len(chain)
            chain.append(env.position)
    return kept,removed


def main():
    runs=read("streets")
    events={r["map"]:r for b in (HERE/"streets-loss-modes.jsonl").read_bytes().splitlines() if (r:=json.loads(b))}
    counts=Counter()
    with (HERE/"lost-loops.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in events:
                continue
            event=events[name]["critical"]
            original=Environment(level)
            trace=runs[name]["actions"]
            kept,removed=shorten(original,trace[:event["step"]-1])
            shortened=Environment(level)
            for char in kept:
                shortened.step(MOVES[int(char)])
            assert (original.position,original.view.memory,original.refuels,original.repairs)==(
                shortened.position,shortened.view.memory,shortened.refuels,shortened.repairs)
            assert shortened.fuel>=original.fuel and shortened.health>=original.health
            state=shortened.position,shortened.fuel,shortened.health
            move=MOVES[int(trace[event["step"]-1])]
            routes=PhysicalRoutes(level)
            restored=transition(routes,state,move,shortened.limit-shortened.steps-1)
            counts[events[name]["mode"],bool(removed),restored]+=1
            row=dict(map=name,split=split,phase=events[name]["mode"],critical=event,loops=removed,
                     removed_steps=sum(removed),saved_fuel=shortened.fuel-original.fuel,
                     saved_health=shortened.health-original.health,original_move_now_viable=restored)
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(counts)


if __name__=="__main__":
    main()
