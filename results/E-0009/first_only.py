"""Дополнительный контроль первой пробы на картах с приобретённым или потерянным финишем."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from run import FILES, HERE, Environment, Level, SightRoute, Status, rows


class FirstProbeControl(SightRoute):
    def __init__(self, only_first):
        super().__init__()
        self.only_first = only_first
        self.offered = 0
        self.executed = 0

    def _select_probe(self, view, service, old, base):
        if self.only_first and self.executed:
            return None
        probe = super()._select_probe(view, service, old, base)
        if probe is None:
            return None
        self.offered += 1
        if not self.only_first and self.offered == 1:
            return None
        self.executed += 1
        return probe


def measure(job):
    series, case, report = job
    level = Level.parse("\n".join(case["lines"] if series == "val" else
                        [case["name"], f'{case["fuel"]} {case["health"]}', *case["grid"]]))
    record = dict(series=series, map=case["name"], base=report["reserve"]["status"],
                  normal=report["sight-route"]["status"])
    first = report["sight-route"]["first"]
    for only_first, name in [(True, "only"), (False, "skip")]:
        bot, env = FirstProbeControl(only_first), Environment(level)
        while env.status == Status.RUNNING:
            move = bot.choose(env.view)
            if env.steps+1 == first["step"]:
                assert bot.offered == 1 and move.name == first["move" if only_first else "base"]
                assert env.position == tuple(first["position"])
            env.step(move)
        assert bot.offered > 0
        record[name] = dict(status=env.status.name, steps=env.steps, probes=bot.executed)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = []
    for series, path in FILES.items():
        episodes, cases = rows(HERE/f"{series}-episodes.jsonl"), rows(path)
        assert len(episodes) == len(cases)
        for ep, case in zip(episodes, cases):
            assert ep["map"] == case["name"]
            agents = ep["agents"]
            if (agents["reserve"]["status"] == "SUCCESS") != (agents["sight-route"]["status"] == "SUCCESS"):
                jobs.append((series, case, agents))
    with args.output.open("x",encoding="utf-8",newline="\n") as f, ProcessPoolExecutor(args.workers) as pool:
        for record in pool.map(measure,jobs):
            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
