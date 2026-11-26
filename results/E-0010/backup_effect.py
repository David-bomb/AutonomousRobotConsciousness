"""Разделить расход запасов и обновление памяти при наблюдательной поездке к тупику."""
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import read
from maze.environment import Environment, Status
from maze.second_window import SecondWindow
from maze.specific_paths import SpecificPaths


def state_at(level, run, step, bot=None):
    env = Environment(level)
    for c in run["actions"][:step]:
        move = MOVES[int(c)]
        if bot is not None:
            assert bot.choose(env.view) == move
        env.step(move)
    return env


def main():
    data = list(map(json.loads, (HERE/"backup-replay.jsonl").read_bytes().splitlines()))
    records = {r["map"]: r for r in data if r["plans"][1]["closed"] and
               r["plans"][1]["actual"].get("backup_visit") and r["plans"][0]["actual"] and
               r["plans"][1]["backup"]["point"] == r["plans"][0]["actual"]["point"]}
    runs = read("specific-paths"), read("second-window")
    outcomes = []
    for split, name, _, _, level in cases():
        if name not in records:
            continue
        clear_episode_caches()
        r = records[name]
        first, second = r["plans"]
        bot = SecondWindow()
        parent = state_at(level, runs[0][name], r["step"]-1+first["actual"]["steps"])
        child = state_at(level, runs[1][name], second["actual"]["backup_visit"]["step"], bot)
        assert parent.position == child.position
        trials = []
        for parent_stock in (False, True):
            for parent_memory in (False, True):
                env, agent = deepcopy(child), deepcopy(bot)
                if parent_stock:
                    env.fuel, env.health = parent.fuel, parent.health
                if parent_memory:
                    env._memory.grid = parent.view.memory
                env._observe()
                while env.status == Status.RUNNING:
                    env.step(agent.choose(env.view))
                if not (parent_stock or parent_memory):
                    assert (env.status.name, env.steps) == (runs[1][name]["status"], runs[1][name]["steps"])
                trials.append(dict(parent_stock=parent_stock, parent_memory=parent_memory, status=env.status.name,
                                   steps_after=env.steps-child.steps))
        continuation = deepcopy(child)
        direct = SpecificPaths()
        direct.__dict__.update(deepcopy(vars(bot)))
        while continuation.status == Status.RUNNING:
            continuation.step(direct.choose(continuation.view))
        outcomes.append(dict(map=name, split=split, verdict=r["verdict"], point=child.position,
                             parent_stock=[parent.fuel,parent.health], child_stock=[child.fuel,child.health],
                             different_cells=sum(a!=b for a,b in zip("".join(parent.view.memory),
                                                                     "".join(child.view.memory))),
                             direct_from_backup=continuation.status.name, trials=trials))
    (HERE/"backup-effect.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in outcomes),
                                            encoding="utf8")
    print(len(outcomes), "сравнений на общей клетке")


if __name__ == "__main__":
    main()
