"""Проверка оценок неизвестных переходов на границе физически обследованной области."""
import json

import numpy as np

from bench import HERE, MOVES, cases
from compare import read
from maze.cell_hypotheses import occupancy
from maze.edge_visibility import joint_chances
from maze.environment import Environment
from maze.frontier_reserve import _lengths
from maze.road_hypotheses import lattice


def forecasts(view, grid):
    board, chance = occupancy(view.memory)
    width = len(view.memory[0])
    edges = lattice(width, len(view.memory))
    u, v, a, b = edges
    free = np.where(board == b"?", chance, board != b"#")
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    free[goal[1] * width + goal[0]] = 1.
    # Истинная геометрия нужна только для оценки уже рассчитанного прогноза.
    observed = np.frombuffer("".join(grid).encode(), dtype="S1") != b"#"
    target = observed[edges].all(axis=0)
    ids = [y * width + x for x, y in _lengths(view.memory, view.position)]
    boundary = (np.isin(u, ids) | np.isin(v, ids)) & (board[edges] == b"?").any(axis=0) & (board[edges] != b"#").all(axis=0)
    diagonal = (a != u) & (b != u)
    predictions = {"minimum": free[edges].min(axis=0), "joint": joint_chances(free, edges)}
    results = []
    for kind, mask in (("прямой", boundary & ~diagonal), ("диагональ", boundary & diagonal)):
        results.append(dict(kind=kind, n=int(mask.sum()), open=int(target[mask].sum()),
                            squared={k: float(np.square(p[mask] - target[mask]).sum()) for k, p in predictions.items()},
                            predicted={k: float(p[mask].sum()) for k, p in predictions.items()}))
    return results


def main():
    runs = read("simple-exit-paths")
    losses = {r["map"]: r["loss"]["step"] for r in map(json.loads, (HERE / "simple-last-waypoints.jsonl").read_bytes().splitlines())
              if r["loss"]["refuels"] and r["loss"]["repairs"]}
    rows = []
    for split, name, _, _, level in cases():
        run = runs[name]
        if run["status"] != "SUCCESS" and name not in losses:
            continue
        env = Environment(level)
        chosen = None
        for c in run["actions"]:
            if env.steps + 1 == losses.get(name):
                chosen = env.view
                break
            old = env.refuels + env.repairs
            env.step(MOVES[int(c)])
            if env.refuels and env.repairs and env.refuels + env.repairs > old:
                chosen = env.view
        if chosen is not None:
            rows.append(dict(map=name, split=split, success=run["status"] == "SUCCESS",
                             scores=forecasts(chosen, level.grid)))
    (HERE / "boundary-forecasts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    for success in (True, False):
        group = [r for r in rows if r["success"] == success]
        for i in (0, 1):
            scores = [r["scores"][i] for r in group]
            n = sum(s["n"] for s in scores)
            print(success, len(group), scores[0]["kind"], n, sum(s["open"] for s in scores)/n,
                  {k: sum(s["squared"][k] for s in scores)/n for k in ("minimum", "joint")})


if __name__ == "__main__":
    main()
