"""У двух сред с одинаковой историей сравнить все ходы, ещё сохраняющие финиш."""
from collections import Counter
from dataclasses import replace
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, read, transition
from maze.compatible_loops import layout
from maze.environment import Environment


def main():
    runs = read("second-window")
    source = map(json.loads,(HERE/"shield-evidence.jsonl").read_bytes().splitlines())
    records = {r["map"]:r for r in source if r["alternatives"][r["parent"]].get("same_history")}
    rows = []
    for split,name,_,_,level in cases():
        if name not in records:
            continue
        r = records[name]
        prefix = runs[name]["actions"][:r["step"]-1]
        env = Environment(level)
        for c in prefix:
            env.step(MOVES[int(c)])
        mask = int(r["alternatives"][r["parent"]]["mask"],16)
        model = layout(env.view.memory,level.goal,mask)
        other = Environment(replace(level,grid=model.grid))
        replay = Environment(level)
        assert replay.view == other.view
        for c in prefix:
            move = MOVES[int(c)]
            assert replay.step(move) == other.step(move)
        routes = PhysicalRoutes(other.level)
        safe = sorted(m.name for m in MOVES if transition(routes,(other.position,other.fuel,other.health),
                                                          m,other.limit-other.steps-1))
        shared = sorted(set(safe) & set(r["safe"]))
        assert r["parent"] in safe and r["parent"] not in r["safe"]
        rows.append(dict(map=name, split=split, step=r["step"], equal_observations=len(prefix)+1,
                         true_safe=r["safe"], hypothetical_safe=safe, common=shared))
    (HERE/"shield-pairs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    print(len(rows),"пар",Counter(bool(r["common"]) for r in rows))


if __name__ == "__main__":
    main()
