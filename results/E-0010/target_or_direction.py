"""Разделяет ошибку выбора потребности и ошибку наведения в момент потери финиша."""
from collections import Counter
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, read
from first_landmark import landmarks
from maze.environment import Environment
from maze.grid import destination


def main():
    runs = read("street-trunk")
    corrections = [json.loads(b) for b in (HERE/"one-correction-street-trunk.jsonl").read_bytes().splitlines()]
    failures = {r["map"]:r for r in corrections}
    assert set(failures) == {k for k,r in runs.items() if r["status"] != "SUCCESS"}
    counts = Counter()
    with (HERE/"target-or-direction.jsonl").open("x", encoding="utf8") as out:
        for split, name, tier, family, level in cases():
            if name not in failures:
                continue
            entry = failures[name]
            env = Environment(level)
            event = entry["critical"]
            for char in runs[name]["actions"][:event["step"]-1]:
                env.step(MOVES[int(char)])
            first = landmarks(env, PhysicalRoutes(level))
            mode = entry["mode"]
            selected = "F" if "заправку" in mode else "R" if "мастерскую" in mode else None
            downhill = []
            if selected:
                field = env._lengths[selected]
                downhill = [m.name for m in MOVES if (q:=destination(level.grid, env.position, m)) in field
                            and field[q] == field[env.position]-1]
            viable = set(event["viable"])
            supported = sorted(viable.intersection(downhill))
            first_kinds = {r["kind"] for r in first}
            counts["states"] += 1
            if selected:
                counts["service_mode"] += 1
                counts["selected_can_be_first"] += selected in first_kinds
                counts["some_safe_downhill"] += bool(supported)
                counts["every_downhill_safe"] += bool(downhill) and len(supported)==len(downhill)
            out.write(json.dumps(dict(map=name,split=split,tier=tier,family=family,critical=event,
                                      mode=mode,selected=selected,first=first,downhill=downhill,
                                      viable_downhill=supported), ensure_ascii=False)+"\n")
            out.flush()
    print(counts)


if __name__ == "__main__":
    main()
