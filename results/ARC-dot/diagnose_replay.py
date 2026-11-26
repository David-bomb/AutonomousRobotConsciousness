"""Проверки утверждений о решающих ходах и о проходимости стресс-карт.

Запуск с путями к стенду E-0005, его результатам E-0004 и E-0005.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stand", type=Path)
    parser.add_argument("e4", type=Path)
    parser.add_argument("e5", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stand.resolve()))
    from maze.automaton import ResourceAutomaton
    from maze.environment import Environment, Status
    from maze.frontier_automaton import FrontierAutomaton, _arrive
    from maze.frontier_reserve import ReserveFrontier, _optimistic, _relaxed_route, _service_floors
    from maze.grid import cell, destination, locations
    from maze.interface import Move
    from maze.levels import Level
    from maze.sarsa import Sarsa, encode
    from scripts.audit_automaton import Reachability
    from scripts.compare_frontier_reserve import fresh
    from scripts.evaluate_frontier import groups, POLICY

    root = Path(__file__).resolve().parent
    cases = {split: {c.name: c for c in batch}
             for split, batch in groups({"val", "archive", "holdout"})}
    data = {}
    e4 = read(args.e4)
    q = read(POLICY)["values"]
    sarsa = []
    for row in e4["failures"]:
        if (row["split"], row["agent"]) != ("val", "sarsa"):
            continue
        env = Environment(cases["val"][row["case"]].level)
        bot = Sarsa(q)
        for _ in range(row["step"] - 1):
            env.step(bot.choose(env.view))
        v = env.view
        scores = {m.name: q[key] for m, key in encode(v).options}
        move = bot.choose(v).name
        assert move == row["move"]
        best_viable = max(scores[name] for name in row["viable"])
        margins = [v.fuel - v.fuel_distance, v.health - v.repair_distance]
        sarsa.append(dict(case=row["case"], step=row["step"], margins=margins,
                          chosen=move, chosen_q=scores[move], best_viable_q=best_viable))
    data["sarsa_val"] = sarsa

    def state(agent):
        return tuple((key, state(value) if key == "_fallback" else
                      tuple(sorted((k.name, n) for k, n in value.items())) if isinstance(value, dict)
                      else value) for key, value in sorted(vars(agent).items()))

    def special(case, bot):
        env = Environment(case.level)
        seen = {}
        repeat = None
        first_fuel = None
        while env.status == Status.RUNNING:
            key = (env.view, state(bot))
            if repeat is None:
                if key in seen:
                    repeat = dict(start=seen[key], period=env.steps-seen[key])
                else:
                    seen[key] = env.steps
            fuels = env.refuels
            v = env.step(bot.choose(env.view))
            if first_fuel is None and env.refuels > fuels:
                first_fuel = dict(step=env.steps, health=v.health, repair_distance=v.repair_distance)
        return dict(outcome=env.status.name, steps=env.steps, cycle=repeat, first_fuel=first_fuel)

    data["resources_archive"] = {
        name: {offset: special(cases["archive"][name], ResourceAutomaton(offset))
               for offset in (0, 2, 3)} for name in ("08-galleries", "09-switchbacks")}
    case = cases["val"]["val-3-10"]
    r = next(x for x in e4["failures"] if x["case"] == case.name and x["agent"] == "resources")
    env, bot = Environment(case.level), ResourceAutomaton()
    for _ in range(r["step"] - 1):
        env.step(bot.choose(env.view))
    before = env.view
    m = bot.choose(before)
    after = env.step(m)
    data["resources_val_3_10"] = dict(step=r["step"], move=m.name, before=[before.fuel,before.fuel_distance],
                                      after=[after.fuel,after.fuel_distance], viable=r["viable"])

    results = []
    for row in e4["failures"]:
        if (row["split"], row["agent"]) != ("holdout", "frontier"):
            continue
        case = cases["holdout"][row["case"]]
        env, bot = Environment(case.level), FrontierAutomaton()
        for _ in range(row["step"]):
            v = env.step(bot.choose(env.view))
        record = dict(case=case.name, step=row["step"],
                      margins=[row["fuel"]-row["fuel_distance"], row["health"]-row["repair_distance"]],
                      after_margins=[v.fuel-v.fuel_distance,v.health-v.repair_distance])
        if row["step"] > 1000:
            record["cycle"] = special(case, FrontierAutomaton())["cycle"]
        results.append(record)
    data["frontier_failures"] = results

    reserve_cases = {c.name: c for c in fresh()}
    results = []
    for row in read(args.e5)["failures"]:
        if row["agent"] != "frontier-reserve":
            continue
        case = reserve_cases[row["case"]]
        env, bot = Environment(case.level), ReserveFrontier()
        for _ in range(row["step"] - 1):
            env.step(bot.choose(env.view))
        v = env.view
        previous = bot._fallback.last_lengths.copy()
        chosen = bot.choose(v)
        assert chosen.name == row["move"]
        targets = tuple((p, k) for k in "FR" for p in locations(v.memory, k)) + ((case.level.goal, "B"),)
        allowed = []
        for m in Move:
            p = destination(v.memory, v.position, m)
            if p is None:
                continue
            kind = "B" if p == case.level.goal else cell(v.memory, *p)
            stocks = _arrive(kind, 1, v.fuel, v.health)
            if stocks and _relaxed_route(_optimistic(v.memory), p, case.level.goal,
                                        *stocks, targets, _service_floors(v, previous, p)):
                allowed.append(m.name)
        results.append(dict(case=case.name, step=row["step"], activity=bot.activity,
                            margins=[v.fuel-v.fuel_distance, v.health-v.repair_distance],
                            chosen=chosen.name, optimistic=allowed, viable=row["viable"]))
    data["reserve_failures"] = results

    maps = [json.loads(line) for line in (root/"source/ARC-stress-100/maps.jsonl").read_bytes().splitlines()]
    checked = []
    for row in maps:
        level = Level.parse("\n".join([row["name"], f'{row["fuel"]} {row["health"]}', *row["grid"]]))
        oracle = Reachability(level)
        optimum = row["optimum"]
        def feasible(limit):
            return oracle.possible(level.start, level.fuel, level.health, limit)
        assert feasible(optimum) and not feasible(optimum-1), row["name"]
        direct = oracle.distance[level.start]
        assert max(level.fuel, level.health) < direct == row["direct"]
        assert row["min_start_slack"] == min(stock-lengths[level.start] for stock, lengths
                                             in zip((level.fuel, level.health), oracle.services))
        checked.append(row["name"])
    data["stress_maps"] = dict(checked_optima=len(checked), families=dict(Counter(r["family"] for r in maps)),
                               median_margin={t: median(r["min_start_slack"] for r in maps if r["tier"] == t)
                                              for t in ("H1", "H2", "H3", "H4")})
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
