"""Фиксация до наблюдения исходов и парные физические поездки только на train."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT)]
import evaluate
from maze.datasets import geometry
from maze.environment import Environment, Status
from maze.interface import Move
from maze.levels import Level
from scripts.audit_automaton import Reachability

TRAIN = ROOT / "results/E-0016/sequence/train.jsonl"
ARCHIVE = ROOT / "results/E-0010/snapshots/pause-mandatory-nov26.zip"
TIERS = ("H1", "H2", "H3", "H4")
MODES = ("full", "previous")
THRESHOLD = 75
MOVES = tuple(Move)


def digest(data):
    return sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def train():
    data = TRAIN.read_bytes()
    if digest(data) != "6affb0aa46384b2f3485e630524bb0dc02abb5b5f9901adee4f786b0d1ada763":
        raise ValueError("Изменился train")
    rows = [(i, json.loads(line)) for i, line in enumerate(data.splitlines()[700:800], 700)]
    assert Counter(r["tier"] for i, r in rows) == dict.fromkeys(TIERS, 25)
    assert len({r["geometry"] for i, r in rows}) == 100
    for i, r in rows:
        assert geometry(r["grid"]) == r["geometry"]
    return dict(rows)


def group(rows, count):
    seen = Counter()
    ids = []
    for index, row in rows.items():
        tier = row["tier"]
        seen[tier] += 1
        if seen[tier] <= count:
            ids.append(index)
    assert len(ids) == len(TIERS) * count
    return ids


def registration():
    # Импорт изолированной точки входа даёт именно её транзитивные модули.
    import subprocess
    expr = ("import json,sys; from maze.mandatory_supply import MandatoryProof; "
            "print(json.dumps([m for m in sys.modules if m.startswith('maze.')]))")
    modules = json.loads(subprocess.check_output([sys.executable, "-B", "-c", expr], cwd=ROOT))
    archive_paths = ["maze/__init__.py", "results/E-0010/bench.py"] + [m.replace(".", "/")+".py" for m in modules]
    sources = {}
    with ZipFile(ARCHIVE) as z:
        for p in sorted(archive_paths):
            content = (ROOT / p).read_bytes().replace(b"\r\n", b"\n")
            assert content == z.read(p).replace(b"\r\n", b"\n"), p
            sources[p] = digest(content)
    inputs = [HERE / "run.py", HERE / "adapter.py", HERE.parent / "policy_worker.py",
              HERE.parent / "evaluate.py", HERE.parent / "policy_api.py", HERE.parent / "suite.py",
              ROOT / "scripts/audit_automaton.py"] + [ROOT / "maze" / n for n in
                                                      ("levels.py", "environment.py", "memory.py", "datasets.py")]
    rows = train()
    return dict(candidate="E-0010/bench.py: mandatory-proof -> maze.mandatory_supply.MandatoryProof",
                archive_sha256=digest(ARCHIVE.read_bytes()), sources_sha256_lf=sources,
                verification_sha256={p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in inputs},
                train=str(TRAIN.relative_to(ROOT).as_posix()), train_sha256=digest(TRAIN.read_bytes()),
                runtime=dict(python=sys.version, numpy=version("numpy"), scipy=version("scipy")),
                selection=dict(slice=[700,800], zero_based=True, order="первые строки каждого H1–H4 в порядке файла",
                               pilot=group(rows,1), slow=group(rows,4), fast=group(rows,8),
                               threshold_seconds=THRESHOLD, workers=4,
                               rule="Четыре пары. Если общее время пробы больше 75 с, взять 4 карты на класс, иначе 8. Проба входит в итог."),
                modes=list(MODES))


def level(row):
    return Level.parse("\n".join([row["name"], f'{row["fuel"]} {row["health"]}', *row["grid"]]))


def state(env):
    return (env.position, env.fuel, env.health, env.refuels, env.repairs)


def loss(level_, limit, states, actions):
    oracle = Reachability(level_)

    def possible(t):
        p, f, h = states[t][:3]
        return oracle.possible(p, f, h, limit-t)

    assert possible(0) and not possible(len(actions))
    left, right = 0, len(actions)
    while right-left > 1:
        mid = (left+right)//2
        if possible(mid):
            left = mid
        else:
            right = mid
    before, after = states[left], states[right]
    p, f, h, refuels, repairs = before
    alternatives = []
    for move, q in oracle.edges[p]:
        nf, nh = oracle.after(q, f, h)
        if oracle.possible(q, nf, nh, limit-right):
            alternatives.append(move.name)
    assert alternatives and actions[left].name not in alternatives
    return dict(step=right, action=actions[left].name, before=before, after=after,
                phase=("FR" if refuels and repairs else "F" if refuels else "R" if repairs else "start"),
                remaining_after=limit-right, alternatives=alternatives)


def episode(job):
    index, row, mode, out = job
    path = out/f"{index}-{mode}.json"
    if path.exists():
        raise FileExistsError(path)
    case = SimpleNamespace(split="sequence/train", name=row["name"], geometry=row["geometry"],
                           scenario=digest(json.dumps([row["grid"],row["fuel"],row["health"]],
                                           separators=(",",":")).encode()), level=level(row))
    actions, states = [], []

    class Recorder(Environment):
        def step(self, move):
            if not states:
                states.append(state(self))
            result = super().step(move)
            states.append(state(self))
            actions.append(move)
            return result

    started = perf_counter()
    evaluate.Environment = Recorder
    try:
        record = evaluate.episode(case, (HERE/"adapter.py").read_bytes(), mode.encode())
    finally:
        evaluate.Environment = Environment
    seconds = perf_counter()-started
    limit = 8*len(case.level.grid)*len(case.level.grid[0])
    numbers = bytes(MOVES.index(m) for m in actions)
    assert record["steps"] == len(numbers) and record["actions_sha256"] == digest(numbers)
    first_loss = loss(case.level, limit, states, actions) if record["outcome"] != "SUCCESS" else None
    write(path, dict(**record, mode=mode, index=index, tier=row["tier"], family=row["family"],
                     initial=[row["fuel"],row["health"]], limit=limit,
                     actions="".join(map(str, numbers)), first_loss=first_loss,
                     elapsed_seconds=seconds, diagnosis_seconds=perf_counter()-started-seconds))
    return index, mode


def run(out):
    lock = read(HERE/"lock.json")
    assert registration() == lock["registration"]
    out.mkdir(exist_ok=False)
    rows = train()
    pilot = group(rows,1)
    (out/"episodes").mkdir()

    def jobs(indices):
        return [(i,rows[i],m,out/"episodes") for i in indices for m in MODES]

    with ProcessPoolExecutor(4) as pool:
        start = perf_counter()
        list(pool.map(episode, jobs(pilot)))
        seconds = perf_counter()-start
        per_tier = 4 if seconds > THRESHOLD else 8
        write(out/"timing-rule.json",dict(probe_seconds=seconds, per_tier=per_tier, pilot=pilot,
                                        lock_sha256=digest((HERE/"lock.json").read_bytes())))
        ids = group(rows,per_tier)
        list(pool.map(episode, jobs(i for i in ids if i not in pilot)))
    write(out/"wall.json",dict(seconds=perf_counter()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register","run"))
    parser.add_argument("--out", type=Path, default=HERE/"physical")
    args = parser.parse_args()
    if args.command == "register":
        write(HERE/"lock.json",dict(utc=datetime.now(timezone.utc).isoformat(), registration=registration()))
    else:
        run(args.out)
