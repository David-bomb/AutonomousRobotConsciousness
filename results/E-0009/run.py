"""Парные физические эпизоды E-0009 и независимый наблюдатель."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARC = HERE.parent / "ARC-dot"
PACK = ARC / "source/ARC-stress-100"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACK))
from maze.datasets import DATA, geometry, load_split
from maze.environment import Environment, Status
from maze.frontier_reserve import ReserveFrontier
from maze.grid import destination
from maze.levels import Level
from maze.pilot_automaton import Mode
from maze.resource_automaton import possible_directions
from maze.sight_automaton import SightRoute, SightDetour
from scripts.audit_automaton import Reachability

SEED = 27110531
FILES = {"known1": PACK/"maps.jsonl", "known2": HERE.parent/"E-0007/fresh.jsonl",
         "fresh": HERE/"fresh.jsonl", "val": DATA/"val.jsonl"}


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def digest(path):
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def generate():
    import make_hard_maps
    used = set(json.loads((PACK/"excluded.json").read_bytes()))
    for path in (FILES["known1"], FILES["known2"]):
        used.update(r["geometry"] for r in rows(path))
    used.update(json.loads((ARC/"replay/e6-fresh/diagnosis.json").read_bytes())["geometries"].values())
    used.update(geometry(c.level.grid) for s in ("train", "val") for c in load_split(s))
    used.update(r["geometry"] for r in json.loads((DATA/"test-manifest.json").read_bytes()).values())
    make_hard_maps.SEED = SEED
    maps = list(make_hard_maps.generate(used))
    hashes = {geometry(m["grid"]) for m in maps}
    assert len(hashes) == len(maps) == 100 and not hashes & used
    with FILES["fresh"].open("x", encoding="utf-8", newline="\n") as f:
        for item in maps:
            item["name"] = "check-" + item["name"]
            f.write(json.dumps(item) + "\n")


def fingerprint():
    archive = json.loads((ARC/"replay/provenance.json").read_bytes())
    locked = archive["base_sources_lf"] | archive["controllers_lf"]["E6"]
    sources = [ROOT/"maze"/f for f in ("environment.py", "memory.py", "grid.py", "interface.py",
               "frontier_automaton.py", "frontier_reserve.py", "pilot_automaton.py", "resource_automaton.py",
               "levels.py")]
    for path in sources:
        assert digest(path) == locked[path.relative_to(ROOT).as_posix()]
    sources.extend([*FILES.values(), Path(__file__), HERE/"protocol.md", ROOT/"maze/sight_automaton.py",
                    PACK/"make_hard_maps.py", ROOT/"scripts/audit_automaton.py"])
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}


class Monitor:
    """Поля наблюдателя используются только после того, как контроллер выбрал ход."""
    reading = None

    def _select_probe(self, view, service, old, base):
        result = super()._select_probe(view, service, old, base)
        if result:
            self.reading = service, old, base, result
        return result


class ObservedRoute(Monitor, SightRoute):
    pass


class ObservedDetour(Monitor, SightDetour):
    pass


def positions(view):
    return dict(position=view.position, fuel=view.fuel, health=view.health,
                dF=view.fuel_distance, dR=view.repair_distance)


def directions(view, service, old):
    return set(possible_directions(view, service, old))


def probe_record(step, before, after, reading, env):
    service, old, default, probe = reading
    original = directions(before, service, old)
    observed = directions(replace(before, memory=after.memory), service, old)
    # Независимо раскрываем лишь окно базового предложения для сравнения двух
    # окон в одном состоянии. Этот снимок не возвращается в контроллер.
    q = destination(before.memory, before.position, default)
    x, y = q
    memory = list(before.memory)
    left, right = max(0, x-1), min(len(memory[0]), x+2)
    for j in range(max(0, y-1), min(len(memory), y+2)):
        memory[j] = memory[j][:left] + env.level.grid[j][left:right] + memory[j][right:]
    alternative = directions(replace(before, memory=tuple(memory)), service, old)
    assert observed and alternative and observed <= original and alternative <= original
    sensor = after.fuel_distance if service.mode == Mode.FUEL else after.repair_distance
    return dict(step=step, **positions(before), service=service.mode.name, base=default.name,
                move=probe.move.name, detour=probe.detour, candidates=len(original),
                removed=sorted(m.name for m in original-observed),
                base_removed=sorted(m.name for m in original-alternative),
                cells=probe.distinguishing_cells, sensor_delta=sensor-service.distance)


def drive(level, bot):
    env = Environment(level)
    track, probes = [], []
    while env.status == Status.RUNNING:
        view = env.view
        action = bot.choose(view)
        activity = bot.activity
        after = env.step(action)
        assert after.moved
        track.append((view, action, activity))
        if isinstance(bot, Monitor) and bot.reading:
            probes.append(probe_record(env.steps, view, after, bot.reading, env))
            bot.reading = None
    outcome = dict(status=env.status.name, steps=env.steps, refuels=env.refuels, repairs=env.repairs)
    return outcome, track, env.view, probes


def next_viable(oracle, state, move, remaining):
    point = destination(oracle.grid, state.position, move)
    fuel, health = oracle.after(point, state.fuel, state.health)
    return oracle.possible(point, fuel, health, remaining)


def critical(oracle, track, last, limit):
    lo, hi = 0, len(track)-1
    while lo < hi:
        mid = (lo+hi)//2
        after = track[mid+1][0]
        if oracle.possible(after.position, after.fuel, after.health, limit-mid-1):
            lo = mid+1
        else:
            hi = mid
    before, move, mode = track[lo]
    after = last if lo == len(track)-1 else track[lo+1][0]
    assert oracle.possible(before.position, before.fuel, before.health, limit-lo)
    assert not oracle.possible(after.position, after.fuel, after.health, limit-lo-1)
    viable = [m.name for m, _ in oracle.edges[before.position] if next_viable(oracle, before, m, limit-lo-1)]
    return dict(step=lo+1, **positions(before), move=move.name, mode=mode, viable=viable)


def measure(item):
    level = Level.parse("\n".join([item["name"], f'{item["fuel"]} {item["health"]}', *item["grid"]]))
    result = dict(map=item["name"], tier=item["tier"], family=item["family"], agents={})
    baseline, base_track, base_end, _ = drive(level, ReserveFrontier())
    result["agents"]["reserve"] = baseline
    oracle = None
    limit = 8*len(level.grid)*len(level.grid[0])
    for name, factory in (("sight-route", ObservedRoute), ("sight-detour", ObservedDetour)):
        data, track, last, probes = drive(level, factory())
        data["probes"] = probes
        result["agents"][name] = data
        index = next((i for i, (a, b) in enumerate(zip(base_track, track)) if a[1] != b[1]), None)
        if index is None:
            assert all(baseline[k] == data[k] for k in ("steps", "status", "refuels", "repairs"))
            assert not probes
            continue
        before, base_move, mode = base_track[index]
        _, move, new_mode = track[index]
        assert before == track[index][0] and probes[0]["step"] == index+1
        if oracle is None:
            oracle = Reachability(level)
        data["first"] = dict(step=index+1, **positions(before), base=base_move.name, move=move.name,
                             mode=mode, new_mode=new_mode,
                             base_viable=next_viable(oracle, before, base_move, limit-index-1),
                             viable=next_viable(oracle, before, move, limit-index-1))
        if baseline["status"] != "SUCCESS" and "critical" not in baseline:
            baseline["critical"] = critical(oracle, base_track, base_end, limit)
        if data["status"] != "SUCCESS":
            data["critical"] = critical(oracle, track, last, limit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("generate", "freeze", *FILES))
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.job == "generate":
        generate()
        return
    frozen = HERE/"frozen.json"
    hashes = fingerprint()
    if args.job == "freeze":
        with frozen.open("x", encoding="utf-8") as f:
            json.dump(dict(seed=SEED, sha256_lf=hashes), f, indent=2)
        return
    assert json.loads(frozen.read_bytes())["sha256_lf"] == hashes
    if not args.output:
        parser.error("Нужен путь для нового журнала")
    cases = rows(FILES[args.job])
    if args.job == "val":
        cases = [dict(name=r["name"], tier=r["stage"], family="val", grid=r["lines"][2:],
                      fuel=r["lines"][1].split()[0], health=r["lines"][1].split()[1]) for r in cases]
    with args.output.open("x", encoding="utf-8", newline="\n") as f, ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(measure, cases), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(args.job, i, flush=True)


if __name__ == "__main__":
    main()
