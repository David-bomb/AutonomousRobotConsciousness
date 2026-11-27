"""Закрепление контроллера, пошаговая оценка и повторная проверка журнала."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from suite import (ARCHIVE, HERE, ROOT, SPLITS, check, checked_suite, digest, file_hash,
                   lf_hash, object_hash, path_name, read, write_new)
from maze.environment import Environment, Status
from maze.interface import Move
from policy_api import ORDER

TOOLS = ("evaluate.py", "suite.py", "policy_worker.py", "policy_api.py")
TERMINAL = {s.name for s in Status if s != Status.RUNNING}
OBS_FIELDS = ("memory", "window", "position", "dx", "dy", "fuel_distance", "repair_distance",
              "fuel", "health", "moved")


def lock_data(adapter, checkpoint, expected, spec, dependencies):
    check(file_hash(checkpoint) == expected, "SHA-256 весов не совпал с объявленным")
    dependencies = set(dependencies)
    runtime = {"python": sys.version}
    adapter_sha = file_hash(adapter)
    if adapter_sha == file_hash(HERE / "e16_adapter.py"):
        dependencies.update(ROOT / "results/E-0016" / f for f in ("policy.py", "geometry.dll", "geometry.cpp"))
    if any(p.name == "geometry.dll" for p in dependencies):
        runtime.update(torch=version("torch"), numpy=version("numpy"))
    return dict(version=1, runtime=runtime, adapter=path_name(adapter), adapter_sha256=adapter_sha,
                checkpoint=path_name(checkpoint), checkpoint_sha256=expected, spec=path_name(spec),
                spec_sha256=object_hash(read(spec)),
                dependencies={path_name(p): file_hash(p) for p in sorted(dependencies)},
                tools_sha256_lf={f: lf_hash(HERE / f) for f in TOOLS})


def checked_lock(path, expected):
    lock = read(path)
    check(object_hash(lock) == expected, "SHA-256 регистрации не совпал")
    actual = lock_data(ROOT / lock["adapter"], ROOT / lock["checkpoint"], lock["checkpoint_sha256"],
                       ROOT / lock["spec"], [ROOT / p for p in lock["dependencies"]])
    check(lock == actual, "Состав или содержимое закреплённых файлов изменились")
    return lock


def episode(case, adapter, weights):
    env = Environment(case.level)
    actions = bytearray()
    system = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
                                         "SYSTEMDRIVE", "USERPROFILE", "LOCALAPPDATA", "APPDATA")
              if k in os.environ}
    cpu = dict(system, CUDA_VISIBLE_DEVICES="-1", PYTHONHASHSEED="0", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (tempfile.TemporaryDirectory() as cwd,
          subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "policy_worker.py")],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=cwd, env=cpu) as worker):
        try:
            for data in (adapter, weights):
                worker.stdin.write(f"{len(data)}\n".encode("ascii"))
                worker.stdin.write(data)
            while env.status == Status.RUNNING:
                view = env.view
                packet = {k: getattr(view, k) for k in OBS_FIELDS}
                packet["last_move"] = view.last_move.name if view.last_move else None
                worker.stdin.write(json.dumps(packet, separators=(",", ":")).encode("utf-8") + b"\n")
                worker.stdin.flush()
                reply = worker.stdout.readline().decode("ascii").rstrip("\r\n")
                check(reply in Move.__members__, "Процесс контроллера не вернул физический ход")
                move = Move[reply]
                env.step(move)
                actions.append(ORDER.index(move))
            worker.stdin.close()
            check(worker.wait() == 0, "Ошибка процесса контроллера")
        finally:
            if worker.poll() is None:
                worker.kill()
    return dict(split=case.split, id=case.name, geometry=case.geometry, scenario_sha256=case.scenario,
                outcome=env.status.name, steps=env.steps, refuels=env.refuels, repairs=env.repairs,
                actions_sha256=digest(actions))


def summarize(cases, rows, weights_sha, groups=SPLITS, threshold=80):
    n = 100 * len(groups)
    check(n and len(cases) == len(rows) == n, f"Нужно ровно {n} поездок")
    check(Counter(c.split for c in cases) == dict.fromkeys(groups, 100)
          and len({c.name for c in cases}) == n, "Неполный реестр сотен")
    wins = dict.fromkeys(groups, 0)
    for c, row in zip(cases, rows, strict=True):
        check((row["split"], row["id"], row["geometry"], row["scenario_sha256"])
              == (c.split, c.name, c.geometry, c.scenario), "Чужая, пропущенная или повторная карта")
        check(row["checkpoint_sha256"] == weights_sha, "В журнале другие веса")
        limit = 8 * len(c.level.grid) * len(c.level.grid[0])
        steps, f, r = (row[k] for k in ("steps", "refuels", "repairs"))
        check(all(type(x) is int for x in (steps, f, r)) and 1 <= steps <= limit
              and min(f, r) >= 0 and f + r <= steps, "Ошибочные счётчики")
        outcome, action_hash = row["outcome"], row["actions_sha256"]
        check(outcome in TERMINAL and (outcome != "LIMIT" or steps == limit), "Ошибочный исход")
        check(len(action_hash) == 64 and all(c in "0123456789abcdef" for c in action_hash),
              "Ошибочный хеш ходов")
        wins[c.split] += outcome == "SUCCESS"
    return dict(episodes=n, successes=wins,
                passed=None if threshold is None else all(x >= threshold for x in wins.values()))


def receipt(cases, out, lock, lock_sha):
    data = (out / "episodes.jsonl").read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    spec = read(ROOT / lock["spec"])
    groups = SPLITS if spec["kind"] == "historical" else spec["contract"]["sets"].keys()
    totals = summarize(cases, rows, lock["checkpoint_sha256"], groups, spec["contract"]["threshold"])
    identity = {k: lock[k] for k in ("adapter_sha256", "checkpoint_sha256", "dependencies",
                                      "tools_sha256_lf", "runtime")}
    return dict(**totals, kind=spec["kind"], controller_sha256=object_hash(identity), checkpoint_sha256=lock["checkpoint_sha256"],
                lock_sha256=lock_sha, spec_sha256=lock["spec_sha256"], episodes_sha256=digest(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--adapter", type=Path, required=True)
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--sha256", required=True)
    freeze.add_argument("--spec", type=Path, default=ARCHIVE)
    freeze.add_argument("--dependency", type=Path, action="append", default=[])
    freeze.add_argument("--out", type=Path, required=True)
    for command in ("run", "verify"):
        p = commands.add_parser(command)
        p.add_argument("--lock", type=Path, required=True)
        p.add_argument("--lock-sha256", required=True)
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        lock = lock_data(args.adapter, args.checkpoint, args.sha256, args.spec, args.dependency)
        checked_suite(args.spec)
        write_new(args.out, lock)
        print("SHA-256 регистрации (канонический JSON)", object_hash(lock))
        return

    lock = checked_lock(args.lock, args.lock_sha256)
    cases = checked_suite(ROOT / lock["spec"])
    if args.command == "run":
        adapter = (ROOT / lock["adapter"]).read_bytes()
        weights = (ROOT / lock["checkpoint"]).read_bytes()
        check(digest(adapter) == lock["adapter_sha256"] and digest(weights) == lock["checkpoint_sha256"],
              "Снимок адаптера или весов не совпал с регистрацией")
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                row = episode(case, adapter, weights)
                row["checkpoint_sha256"] = lock["checkpoint_sha256"]
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
        checked_lock(args.lock, args.lock_sha256)
    result = receipt(cases, args.out, lock, args.lock_sha256)
    if args.command == "run":
        write_new(args.out / "summary.json", result)
    else:
        check(result == read(args.out / "summary.json"), "Сводка или файл поездок изменились")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
