"""Подмена одной команды в точке первой потери достижимости, затем штатная политика."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from audit import E7, HERE, PACK, Environment, Level, ReserveFrontier, Status
from maze.interface import Move


def replay(job):
    case, event, alternative = job
    level = Level.parse("\n".join([case["name"], f'{case["fuel"]} {case["health"]}', *case["grid"]]))
    env, bot = Environment(level), ReserveFrontier()
    while env.status == Status.RUNNING:
        move = bot.choose(env.view)
        if env.steps + 1 == event["step"]:
            assert move.name == event["move"]
            assert list(env.position) == event["position"]
            assert (env.fuel, env.health) == (event["fuel"], event["health"])
            move = Move[alternative]
        env.step(move)
        assert env.moved
    assert env.steps >= event["step"]
    return dict(map=case["name"], step=event["step"], chosen=event["move"],
                replacement=alternative, outcome=env.status.name, steps=env.steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    maps = {}
    for path in (PACK/"maps.jsonl", E7/"fresh.jsonl"):
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            maps[row["name"]] = row
    jobs = []
    for line in (HERE/"trace.jsonl").read_bytes().splitlines():
        record = json.loads(line)
        event = record["critical"]
        jobs.extend((maps[record["map"]], event, alt) for alt in event["viable"])
    assert len({case["name"] for case, _, _ in jobs}) == 90
    with args.output.open("x", encoding="utf-8", newline="\n") as stream, ProcessPoolExecutor(args.workers) as pool:
        for record in pool.map(replay, jobs):
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
