"""После возврата проверить, не выбран ли выход, уже исключённый прежней поездкой."""
from collections import Counter
import json

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment
from maze.grid import destination
from maze.necessary_visit import necessary_visit
from maze.sight_automaton import window_unknown


def main():
    controls = {}
    for row in map(json.loads, (HERE/"return-controls.jsonl").read_bytes().splitlines()):
        controls.setdefault(row["map"], {})[row["step"]] = row
    runs, records = read("specific-paths"), []
    for split, name, _, _, level in cases():
        if name not in controls:
            continue
        env = Environment(level)
        observed = {}
        for step, char in enumerate(runs[name]["actions"], 1):
            v = env.view
            observed[v.position] = v.fuel_distance, v.repair_distance
            if step in controls[name]:
                row = controls[name][step]
                choices = {}
                for key, direction in row["choices"].items():
                    q = destination(v.memory, v.position, next(m for m in MOVES if m.name == direction))
                    required = necessary_visit(v.memory, v.position, q)
                    assert not required or q in observed
                    choices[key] = dict(move=direction, at=observed.get(q), unseen=len(window_unknown(v.memory,q)), required=required,
                                        safe=direction in row["safe"])
                records.append(dict(map=name, split=split, step=step, here=observed[v.position],
                                    stocks=[v.fuel,v.health], choices=choices, safe=row["safe"]))
            if step >= max(controls[name]):
                break
            env.step(MOVES[int(char)])
    assert len(records) == sum(map(len, controls.values()))
    (HERE/"returned-choices.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf8")
    for key in ("specific-paths","last-return-paths"):
        print(key, Counter((c["at"] is not None, c["safe"]) for r in records if (c:=r["choices"][key])))


if __name__ == "__main__":
    main()
