"""Разбор стресс-карт без единого финиша. Аргументы: стенд E-0006 и новый JSON."""
import csv
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Использование: diagnose_unsolved.py <стенд E-0006> <новый JSON>")
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from maze.environment import Environment, Status
from maze.frontier_automaton import _arrive
from maze.frontier_reserve import ReserveFrontier, _optimistic, _relaxed_route, _service_floors
from maze.grid import cell, destination, locations
from maze.interface import Move
from maze.levels import Level
from maze.pilot_automaton import Mode
from scripts.audit_automaton import Reachability
from scripts.evaluate_frontier import Transition, first_irreversible


def shortest_route(oracle, start, fuel, health, limit):
    """Поиск в ширину с восстановлением ходов. Доминирующий запас достигнут не позже."""
    origin = (start, fuel, health)
    queue = deque([(origin, 0)])
    parents = {origin: None}
    seen = {start: [(fuel, health)]}
    while queue:
        state, length = queue.popleft()
        point, f, h = state
        if point == oracle.goal:
            moves = []
            while parents[state] is not None:
                state, move = parents[state]
                moves.append(move)
            return moves[::-1]
        distance = oracle.distance[point]
        if min(f, h) <= 0 or length + distance > limit:
            continue
        if any(stock < min(distance, paths[point]) for stock, paths in zip((f, h), oracle.services)):
            continue
        for move, q in oracle.edges[point]:
            nf, nh = oracle.after(q, f, h)
            stocks = seen.setdefault(q, [])
            if any(old_f >= nf and old_h >= nh for old_f, old_h in stocks):
                continue
            stocks.append((nf, nh))
            nxt = (q, nf, nh)
            parents[nxt] = (state, move)
            queue.append((nxt, length + 1))
    raise ValueError("Нет маршрута в заданном бюджете")


def checked_drive(level, moves):
    env = Environment(level)
    services = []
    for move in moves:
        v = env.step(move)
        assert v.moved
        kind = cell(level.grid, *v.position)
        if kind in "FR":
            services.append((env.steps, kind, v.position))
    assert env.status == Status.SUCCESS
    return dict(length=env.steps, services=services)


def anchored_distances(grid, point, targets, samples):
    """При проходе через измеренную клетку прибавить её известную длину до сервиса."""
    queue = deque([(point, 0)])
    seen = {point}
    best = {kind: float("inf") for kind in "FR"}
    while queue:
        q, length = queue.popleft()
        if q in samples:
            for kind, distance in zip("FR", samples[q]):
                best[kind] = min(best[kind], length + distance)
            continue
        if q in targets:
            kind = targets[q]
            best[kind] = min(best[kind], length)
        if length >= max(best.values()):
            break
        for move in Move:
            nxt = destination(grid, q, move)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, length + 1))
    return best


def assess_moves(view, previous, level, oracle, remaining):
    optimistic = _optimistic(view.memory)
    services = {p: k for k in "FR" for p in locations(view.memory, k)}
    targets = (*services.items(), (level.goal, "B"))
    samples = {view.position: (view.fuel_distance, view.repair_distance)}
    old = {}
    if previous is not None:
        samples[previous.position] = (previous.fuel_distance, previous.repair_distance)
        old = {Mode.FUEL: previous.fuel_distance, Mode.REPAIR: previous.repair_distance}
    assessed = {}
    for move in Move:
        q = destination(view.memory, view.position, move)
        if q is None:
            continue
        assert destination(level.grid, view.position, move) == q
        f, h = oracle.after(q, view.fuel, view.health)
        stocks = _arrive("B" if q == level.goal else cell(view.memory, *q), 1, view.fuel, view.health)
        floors = _service_floors(view, old, q)
        anchored = anchored_distances(optimistic, q, services, samples)
        strengthened = {k: max(anchored[k], floors[k]) for k in "FR"}
        simple = bool(stocks and _relaxed_route(optimistic, q, level.goal, *stocks, targets, floors))
        stronger = bool(stocks and _relaxed_route(optimistic, q, level.goal, *stocks, targets, strengthened))
        possible = oracle.possible(q, f, h, remaining)
        assert not possible or (simple and stronger)
        assessed[move.name] = dict(viable=possible, optimistic=simple, anchored=stronger,
                                   fuel=f, health=h, dF=oracle.services[0][q], dR=oracle.services[1][q])
    return assessed


def diagnose(row, expected):
    level = Level.parse("\n".join([row["name"], f'{row["fuel"]} {row["health"]}', *row["grid"]]))
    env, bot = Environment(level), ReserveFrontier()
    oracle = Reachability(level)
    moves = shortest_route(oracle, level.start, level.fuel, level.health, row["optimum"])
    witness = checked_drive(level, moves)
    assert witness["length"] == row["optimum"] and not oracle.possible(
        level.start, level.fuel, level.health, row["optimum"]-1)
    witness["moves"] = [m.name for m in moves]
    history, seen = [], {}
    cycle = None
    while env.status == Status.RUNNING:
        v = env.view
        if expected["status"] == "LIMIT" and cycle is None:
            # У frontier-reserve это скалярные поля и вложенный ресурсный автомат.
            fallback = tuple((k, tuple(value.items()) if isinstance(value, dict) else value)
                             for k, value in vars(bot._fallback).items())
            key = (v, tuple((k,value) for k,value in vars(bot).items() if k != "_fallback"), fallback)
            if key in seen:
                start, fuels, repairs = seen[key]
                cycle = dict(start=start, period=env.steps-start,
                             refuels=env.refuels-fuels, repairs=env.repairs-repairs)
            else:
                seen[key] = (env.steps, env.refuels, env.repairs)
        move = bot.choose(v)
        history.append(Transition(v, move, env.step(move), bot.activity))
    assert (env.status.name, env.steps, env.refuels, env.repairs) == (
        expected["status"], int(expected["steps"]), int(expected["refuels"]), int(expected["repairs"]))
    failure = first_irreversible(oracle, env.limit, history)
    assert failure["initially_feasible"]
    index = failure["step"] - 1
    event = history[index]
    v, after = event.before, event.after
    remaining = env.limit - index - 1
    assessed = assess_moves(v, history[index-1].before if index else None, level, oracle, remaining)
    viable = [name for name, data in assessed.items() if data["viable"]]
    assert viable and viable == failure["viable"] and event.move.name not in viable
    assert not oracle.possible(after.position, after.fuel, after.health, remaining)

    rescue = shortest_route(oracle, v.position, v.fuel, v.health, remaining+1)
    assert rescue[0].name in viable
    prefix = [t.move for t in history[:index]]
    checked = checked_drive(level, prefix + rescue)
    failure.update(options=assessed, window=v.window, memory=v.memory, after_position=after.position,
                   viable_with_fresh_horizon=oracle.possible(after.position, after.fuel, after.health, env.limit),
                   remaining_after=remaining, rescue=[m.name for m in rescue], rescue_total=checked["length"])
    return dict(map=row["name"], family=row["family"], outcome=env.status.name, steps=env.steps,
                witness=witness, prefix=[m.name for m in prefix], critical=failure, cycle=cycle)


def main(stand, output):
    provenance = json.loads((ROOT/"replay/provenance.json").read_text(encoding="utf-8"))
    for name, digest in (provenance["base_sources_lf"] | provenance["controllers_lf"]["E6"]).items():
        assert hashlib.sha256((stand/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() == digest
    source = (ROOT/"source/ARC-stress-100/maps.jsonl").read_bytes()
    assert hashlib.sha256(source).hexdigest() == provenance["stress_maps"]
    maps = [json.loads(line) for line in source.splitlines()]
    with (ROOT/"replay/stress/episodes.csv").open(encoding="utf-8", newline="") as f:
        episodes = list(csv.DictReader(f))
    agents = {r["variant"] for r in episodes}
    assert len(agents) == 16 and len(maps) == 100 and len(episodes) == 1600
    assert {(r["map"],r["variant"]) for r in episodes} == {(m["name"],a) for m in maps for a in agents}
    solved = {r["map"] for r in episodes if r["status"] == "SUCCESS"}
    expected = {r["map"]:r for r in episodes if r["variant"] == "frontier-reserve"}
    results = [diagnose(m, expected[m["name"]]) for m in maps if m["name"] not in solved]
    assert len(results) == 36
    with output.open("x", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
