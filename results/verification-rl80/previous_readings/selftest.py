"""Проверить реальный worker и Environment на полностью искусственных эпизодах."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryFile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import evaluate
from maze.environment import Environment
from maze.interface import Move
from suite import Case, Level, check, digest, geometry, object_hash, read, write_new
from controller import PreviousReadings

ADAPTER = (HERE / "probe_adapter.py").read_bytes()
MOVES = tuple(Move)


def case(name, grid, fuel=2, health=2):
    level = Level.parse("\n".join([name, f"{fuel} {health}", *grid]))
    return Case("synthetic", name, geometry(grid), object_hash([grid, fuel, health]), level)


def distances(view):
    return view.fuel_distance, view.repair_distance


def trace(c, mode):
    # Наблюдатель только записывает вызовы с обеих сторон неизменённого канала.
    # Файл диагностики доступен политике только как выход stderr.
    physical = []
    step = Environment.step
    popen = evaluate.subprocess.Popen

    def record(env, action):
        physical.append((env.view, action))
        return step(env, action)

    with TemporaryFile() as diagnostics:
        def child(*args, **kwargs):
            return popen(*args, **kwargs, stderr=diagnostics)

        with patch.object(evaluate.subprocess, "Popen", child), patch.object(Environment, "step", record):
            row = evaluate.episode(c, ADAPTER, mode)
        diagnostics.seek(0)
        calls = [json.loads(line) for line in diagnostics]
    check(len(physical) == len(calls) == row["steps"], "Числа решений и физических шагов различаются")
    previous = (None, None)
    for (view, _), inputs in zip(physical, calls, strict=True):
        current = distances(view)
        check(inputs == [*previous, *current], "Ошибочные входы решения или момент записи регистров")
        previous = current
    check(row["actions_sha256"] == digest(bytes(MOVES.index(m) for _, m in physical)), "Расходится хеш ходов")
    return row, physical, calls


def pairs():
    key_view = None
    trials = []
    alternatives = {(False, False): Move.EAST, (True, False): Move.NORTH,
                    (False, True): Move.SOUTH, (True, True): Move.WEST}
    for (block_f, block_r), wanted in alternatives.items():
        grid = ["#########", "#....F..#", "#.......#", "#.......#", "#..A.B..#",
                "#.......#", "#.......#", "#....R..#", "#########"]
        for index, blocked in [(2, block_f), (6, block_r)]:
            if blocked:
                grid[index] = "#...#...#"
        c = case(f"pair-{int(block_f)}-{int(block_r)}", grid)
        row, steps, calls = trace(c, b"pair")
        check(row["steps"] == 2 and steps[0][1] == Move.EAST, "Ожидались два физических хода")
        view, move = steps[1]
        if key_view is None:
            key_view = view
        check(view == key_view and distances(view) == (3, 3), "Текущие Observation должны совпадать полностью")
        check(calls[1][:2] == [3+block_f, 3+block_r] and move == wanted, "Прежнее показание потеряно")
        trials.append(dict(previous=calls[1][:2], current=calls[1][2:], move=move.name, outcome=row["outcome"]))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    trials = pairs()

    ring = case("cycle", ["######", "#AFRB#", "######"])
    cycle, records, calls = trace(ring, b"cycle")
    frequencies = Counter(v for v, _ in records)
    check(cycle["outcome"] == "LIMIT" and len(calls) == 8*6*3 and max(frequencies.values()) > 1,
          "Не удалось проверить повтор наблюдения до физического лимита")
    check(any(c[0] == 0 for c in calls) and any(c[1] == 0 for c in calls), "Нужно сохранить нулевые длины")

    single = case("collision", ["######", "#AF.B#", "######"], 4, 4)
    collision, records, calls = trace(single, b"collision")
    before, after = records[1][0], records[2][0]
    check(after.moved is False and before.position == after.position and calls[2] == [0, None, 0, None]
          and collision["outcome"] == "SUCCESS", "Столкновение или отсутствующий ремонт переданы неверно")

    observed = []
    def watch(view, f, r):
        observed.append((f, r))
        return Move.EAST
    ctl = PreviousReadings(watch)
    env = Environment(ring.level)
    start_readings = distances(env.view)
    env.step(ctl.choose(env.view))
    env.step(ctl.choose(env.view))
    ctl.reset()
    env.reset()
    env.step(ctl.choose(env.view))
    check(observed == [(None, None), start_readings, (None, None)], "Явный сброс не очистил оба регистра")
    result = dict(pairs=trials, identical_complete_observation=True,
                  cycle=dict(steps=cycle["steps"], decisions=cycle["steps"], outcome=cycle["outcome"],
                             distinct_observations=len(frequencies), most_visits=max(frequencies.values())),
                  collision=dict(steps=collision["steps"], inputs_after_collision=calls[2],
                                 outcome=collision["outcome"]), explicit_reset=True,
                  worker_processes=len(trials)+2, explicit_reset_steps=3,
                  total_physical_steps=2*len(trials)+cycle["steps"]+collision["steps"]+3)
    target = HERE / "selftest.json"
    if args.write:
        write_new(target, result)
    else:
        check(read(target) == result, "Результат отличается от сохранённой самопроверки")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
