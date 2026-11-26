"""Оценить прогнозы закрытого окна на одном общем наблюдении каждого позднего отказа."""
import json

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment
from maze.first_window import window_arrivals
from maze.grid import destination
from maze.second_window import closed_probability
from maze.window_patterns import forecast, predicted_closure


def main():
    runs = read("return-windows")
    late = {r["map"]:r["first"] for r in map(json.loads,(HERE/"return-windows-next-loss.jsonl").read_bytes().splitlines())}
    rows = []
    for split,name,_,_,level in cases():
        if name not in late:
            continue
        env = Environment(level)
        for c in runs[name]["actions"][:late[name]-1]:
            env.step(MOVES[int(c)])
        v = env.view
        points = sorted({a.point for a in window_arrivals(v)})
        forecasts = []
        for p in points:
            closed = all(destination(level.grid,p,m)==destination(v.memory,p,m) for m in MOVES)
            baseline = closed_probability(v.memory,p,level.goal)
            revised = predicted_closure(v.memory,p,level.goal)
            forecasts.append(dict(point=p,closed=closed,parent=float(baseline),pattern=float(revised),
                                  examples=forecast(v.memory,p,level.goal)[0]))
        rows.append(dict(map=name,split=split,step=late[name],windows=forecasts))
    (HERE/"pattern-forecasts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    for k in ("parent","pattern"):
        examples = [e for r in rows for e in r["windows"]]
        print(k,len(examples),sum((e[k]-e["closed"])**2 for e in examples)/len(examples))


if __name__=="__main__":
    main()
