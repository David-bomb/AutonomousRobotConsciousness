"""Внешняя проверка, какую станцию обозначает показание, на двух фиксированных выборках наблюдений."""
from collections import Counter
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import read
from maze.environment import Environment
from maze.frontier_reserve import _lengths, _optimistic
from maze.grid import locations
from maze.nearest_station import station_weights


def indexed(name):
    return {r["map"]: r for r in map(json.loads, (HERE / name).read_bytes().splitlines())}


def main():
    runs = read("street-trunk")
    ambiguous = indexed("first-ambiguity.jsonl")
    critical = indexed("target-or-direction.jsonl")
    result = []
    for split, name, _, _, level in cases():
        clear_episode_caches()
        targets = []
        event = ambiguous[name]["event"]
        if event:
            targets.append(("первый выбор", event["step"], event["selected"]))
        failed = critical.get(name)
        if failed and failed["selected"] in ("F", "R") and not (
                failed["critical"]["refuels"] and failed["critical"]["repairs"]):
            targets.append(("перед отказом", failed["critical"]["step"], failed["selected"]))
        if not targets:
            continue
        env = Environment(level)
        old = (None, None)
        for step in range(1, 1 + max(t[1] for t in targets)):
            v = env.view
            for cohort, when, kind in targets:
                if when != step:
                    continue
                k = "FR".index(kind)
                d = (v.fuel_distance, v.repair_distance)[k]
                lower = _lengths(_optimistic(v.memory), v.position)
                options = tuple(q for q in locations(v.memory, kind) if lower.get(q, float("inf")) <= d)
                truth = _lengths(level.grid, v.position)
                nearest = [q for q in options if truth[q] == d]
                assert nearest
                goal = v.position[0] + v.dx, v.position[1] + v.dy
                weights = station_weights(v, goal, old if v.moved else (None, None), k, options)
                high = max(weights.values(), default=0)
                best_low = min(lower[q] for q in options)
                result.append(dict(map=name, split=split, step=step, cohort=cohort,
                                   success=runs[name]["status"] == "SUCCESS", kind=kind, distance=d,
                                   stations=[dict(point=q, actual=truth[q], lower=lower[q],
                                                  weight=weights.get(q, 0.)) for q in options],
                                   geometric=[q for q in options if lower[q] == best_low],
                                   predicted=[q for q in options if high and weights[q] == high],
                                   nearest=nearest))
            old = v.fuel_distance, v.repair_distance
            env.step(MOVES[int(runs[name]["actions"][step-1])])
    out = HERE / "station-identity.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf8")
    for cohort in ("первый выбор", "перед отказом"):
        group = [r for r in result if r["cohort"] == cohort]
        print(cohort, len(group), Counter(len(r["stations"]) for r in group))
        for key in ("geometric", "predicted"):
            counts = Counter()
            for r in group:
                if len(r["stations"]) < 2:
                    continue
                choice = set(r[key])
                outcome = ("нет оценки" if not choice else "верно" if choice <= set(r["nearest"])
                           else "равенство" if choice.intersection(r["nearest"]) else "ошибка")
                counts[r["success"], outcome] += 1
            print(key, counts)


if __name__ == "__main__":
    main()
