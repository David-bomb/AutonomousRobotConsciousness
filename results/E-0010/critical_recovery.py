"""Проверить, каких отказов можно избежать одной заменой команды исходного автомата."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import read
from maze.environment import Environment, Status


def measure(job):
    agent_name, case, run, loss = job
    split, name, _, _, level = case
    clear_episode_caches()
    env, bot = Environment(level), controller(agent_name)
    for c in run["actions"][:loss["step"]-1]:
        move = bot.choose(env.view)
        assert move == MOVES[int(c)], name
        env.step(move)
    proposed = bot.choose(env.view)
    assert proposed.name == loss["move"]
    results = []
    for alternative in loss["viable"]:
        episode, machine = deepcopy(env), deepcopy(bot)
        mv = next(m for m in MOVES if m.name == alternative)
        episode.step(mv)
        next_readings = episode.view.fuel_distance, episode.view.repair_distance
        while episode.status == Status.RUNNING:
            episode.step(machine.choose(episode.view))
        results.append(dict(move=alternative, status=episode.status.name, steps=episode.steps, after=next_readings))
    return dict(split=split, map=name, first_loss=loss["step"], proposed=proposed.name,
                serviced=[env.refuels>0,env.repairs>0], trials=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    agent_name = "return-windows"
    runs = read(agent_name)
    comparison = map(json.loads,(HERE/"second-window--return-windows.jsonl").read_bytes().splitlines())
    losses = {r["map"]:r["child_loss"] for r in comparison if "child_loss" in r}
    path = HERE/"return-windows-recovery.jsonl"
    saved = {r["map"] for r in map(json.loads,path.read_bytes().splitlines())} if path.exists() else set()
    assert saved <= losses.keys()
    jobs = [(agent_name,c,runs[c[1]],losses[c[1]]) for c in cases() if c[1] in losses and c[1] not in saved]
    with ProcessPoolExecutor(args.workers) as pool, path.open("a",encoding="utf8") as out:
        for row in pool.map(measure,jobs):
            out.write(json.dumps(row)+"\n")
            out.flush()
    collected = list(map(json.loads,path.read_bytes().splitlines()))
    assert len(collected) == len(losses) == len({r["map"] for r in collected})
    print(sum(any(a["status"]=="SUCCESS" for a in r["trials"]) for r in collected), "из", len(losses))


if __name__ == "__main__":
    main()
