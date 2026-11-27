"""Воздействие признаков на первый необратимый ход базы на новых учебных геометриях."""
from collections import Counter

import numpy as np
import torch

from maze.environment import Environment, Status
from ....data import digest, level, read, rows, save
from ....policy import MOVES
from ....teacher import Teacher
from .cells import HERE
from .model import RoomPolicy
from .physical import ROOT


@torch.inference_mode()
def main():
    out = ROOT / "runs/rooms3000"
    path = out / "rl.pt"
    report = read(out / "independent.json")
    assert digest(path) == report["config"]["weights"]
    torch.set_num_threads(1)
    pack = torch.load(path, weights_only=True, map_location="cpu")
    policy = RoomPolicy(pack)
    weights = pack["linear"].numpy()
    bank = {r["name"]: r for r in rows(HERE / "independent.jsonl")}
    metrics = {}
    for record in report["episodes"]:
        probe = record["tests"].get("first_loss")
        if not probe:
            continue
        env = Environment(level(bank[record["map"]]))
        for _ in range(probe["step"]):
            assert env.status == Status.RUNNING
            z, ok = policy.directions(env.view)
            base = int(np.where(ok, z[:, 0], -np.inf).argmax())
            if env.steps + 1 == probe["step"]:
                break
            env.step(MOVES[base])
        teacher = Teacher(env.level)
        _, safe = teacher.labels(env)
        teacher.remaining.cache_clear()
        picked = int(np.where(ok, z @ weights, -np.inf).argmax())
        assert not safe[base] and safe.any()
        assert bool(safe[picked]) == probe["candidate_safe"] and (picked != base) == probe["changed"]
        delta = z[:, 1:] @ weights[1:]
        changes_rank = bool(np.ptp(delta[ok]) > 1e-9)
        # Сравниваем максимальную прибавку у жизнеспособных и необратимых ходов.
        safe_advantage = delta[safe].max() - delta[ok & ~safe].max()
        for key in {"all", record["family"], record["tier"], record["tier"] + "_" + record["family"]}:
            stats = metrics.setdefault(key, Counter())
            stats.update(dict(
                losses=1, room_classified=int(probe["family_pred"] == 3),
                core_present=int(probe["inferred"] > 0), walls_present=int(probe["walls"] > 0),
                direction_dependent=int(changes_rank), raises_safe_above_other=int(safe_advantage > 1e-9),
                changed=int(picked != base), rescued=int(safe[picked]),
            ))
    save(out / "loss-probe.json", dict(weights=digest(path), source=digest(HERE / "loss_probe.py"),
         independent=digest(out / "independent.json"), groups=metrics))
    print(metrics.get("all"), metrics.get("H4_rooms"), flush=True)


if __name__ == "__main__":
    main()
