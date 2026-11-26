"""Сменить контроллер на одну команду в первом неоднозначном наблюдении родителя."""
from collections import Counter
from copy import deepcopy
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import read
from maze.environment import Environment


MODELS = ("street-trunk", "smallest-detour", "gauge-detour", "street-trunk-reserve", "leg-survival")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--services", action="store_true", help="сравнить распознавание конкретной станции и подхода")
    args = parser.parse_args()
    models = ("street-trunk", "mission-order-reserve", "nearest-station", "service-contact") if args.services else MODELS
    observations = [json.loads(line) for line in (HERE / "first-ambiguity.jsonl").read_bytes().splitlines()]
    records = {r["map"]: r for r in observations}
    runs = read("street-trunk")
    assert len(records) == len(observations) == 336 and records.keys() == runs.keys()
    path = HERE / ("service-ambiguity-controls.jsonl" if args.services else "ambiguity-controls.jsonl")
    saved = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    done = {r["map"] for r in saved}
    assert len(done) == len(saved) and done <= {k for k, r in records.items() if r["event"]}
    with path.open("a", encoding="utf8") as out:
        for split, name, _, _, level in cases():
            event = records[name]["event"]
            if event is None or name in done:
                continue
            clear_episode_caches()
            env = Environment(level)
            parent = controller("street-trunk")
            for c in runs[name]["actions"][:event["step"] - 1]:
                action = parent.choose(env.view)
                assert action == MOVES[int(c)]
                env.step(action)
            view = env.view
            assert list(view.position) == event["pos"]
            choices = {}
            for key in models:
                agent = controller(key)
                agent.__dict__.update(deepcopy(vars(parent)))
                choices[key] = agent.choose(view).name
            assert choices["street-trunk"] == event["choice"]["actual"]
            row = dict(map=name, split=split, step=event["step"], parent_status=runs[name]["status"],
                       selected=event["selected"], safe=event["safe"], choices=choices)
            out.write(json.dumps(row) + "\n")
            out.flush()
            saved.append(row)
    assert len(saved) == sum(r["event"] is not None for r in records.values())
    for key in models:
        counts = Counter((r["parent_status"] == "SUCCESS", r["choices"][key] in r["safe"]) for r in saved)
        changes = Counter((r["choices"]["street-trunk"] in r["safe"], r["choices"][key] in r["safe"]) for r in saved)
        print(key, counts, "приобретены/потеряны", changes[False, True], changes[True, False], flush=True)


if __name__ == "__main__":
    main()
