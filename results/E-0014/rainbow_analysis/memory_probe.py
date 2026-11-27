"""Парная проверка вклада накопленной карты в действия и финиши Rainbow."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from maze.grid import locations
from maze.memory import CellMemory
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import encode
from neural_rl.rollout import MapPool, evaluate
from neural_rl.stress_hundreds import maps


def local_only(image):
    height, width = image.shape[-2:]
    at = (image & 64).flatten(1).argmax(1)
    xs = torch.arange(width, device=image.device)[None, None, :]
    ys = torch.arange(height, device=image.device)[None, :, None]
    near = ((xs - (at % width)[:, None, None]).abs() <= 1) & ((ys - (at // width)[:, None, None]).abs() <= 1)
    preserve = near | ((image & 24) != 0) | (image == 0)
    return torch.where(preserve, image, 1 | (image & 32))


def audit_encoder(pool):
    for case in pool.cases[::19]:
        env = pool.start(case)
        for _ in range(5):
            seen = env.view
            services = {p: k for k in "FR" for p in locations(seen.memory, k)}
            mem = CellMemory(len(seen.memory[0]), len(seen.memory), services)
            alt = replace(seen, memory=mem.update(seen.position, seen.window))
            image, nums, mask = batch([encode(seen, False)])
            fresh, new_nums, new_mask = batch([encode(alt, False)])
            assert torch.equal(local_only(image), fresh) and torch.equal(nums, new_nums) and torch.equal(mask, new_mask)
            if env.status.name != "RUNNING":
                break
            from neural_rl.perception import MOVES
            env.step(MOVES[int(torch.nonzero(mask[0])[0])])


class Probe(Network):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.counts = Counter()
        self.with_history = True

    def forward(self, data):
        image, numbers, mask = data
        fresh = local_only(image)
        other = super().forward((fresh, numbers, mask))
        if not self.with_history:
            return other
        normal = super().forward(data)
        changed = normal[0].argmax(1) != other[0].argmax(1)
        eligible = (mask.sum(1) >= 2) & (image != fresh).flatten(1).any(1)
        self.counts.update(forward_decisions=len(image), eligible=int(eligible.sum()),
                           changed=int((changed & eligible).sum()), unexpected=int((changed & ~eligible).sum()))
        return normal


def run_score(net, pool):
    return [r for start in range(0, len(pool.cases), 32)
            for r in evaluate(net, Chunk(pool, pool.cases[start:start + 32]), 1)]


def main():
    old = HERE.parents[1] / "E-0012"
    stress = HERE.parents[1] / "E-0013"
    manifest = read(stress / "manifest.json")["models"]
    val = Pool("validation_id", 60)
    hundreds = MapPool([case for _, case in maps()])
    setup(1)
    audit_encoder(val)
    summary = {}
    for name in ("rainbow-v03-s4", "rainbow-v01-s4", "rainbow-v03-s4-diag"):
        model = manifest[name]
        weights = HERE.parents[2] / model["weights"]
        assert digest(weights) == model["sha256"]
        net = Probe(model["config"]).cuda()
        net.load_state_dict(torch.load(weights, weights_only=True, map_location="cuda"))
        summary[name] = {}
        for label, pool in (("validation", val), ("hundreds", hundreds)):
            net.with_history = True
            net.counts.clear()
            normal = {r["map"]: r for r in run_score(net, pool)}
            if label == "hundreds":
                base = read(stress / "models" / f"{name}.json")["rows"]
                assert all(normal[r["map"]] == {k:v for k,v in r.items() if k != "split"} for r in base)
            elif model["kind"] == "primary":
                base = read(old / "runs" / name / "validation.json")["after"]
                assert all(normal[r["map"]] == r for r in base)
            net.with_history = False
            counter = {r["map"]: r for r in run_score(net, pool)}
            assert normal.keys() == counter.keys() and net.counts["unexpected"] == 0
            base_wins = {k for k,v in normal.items() if v["outcome"]=="SUCCESS"}
            without_wins = {k for k,v in counter.items() if v["outcome"]=="SUCCESS"}
            summary[name][label] = dict(maps=len(normal), **net.counts, wins=len(base_wins),
                wins_local_only=len(without_wins), gained=sorted(without_wins-base_wins),
                lost=sorted(base_wins-without_wins),
                changed_outcome=sum(counter[k]["outcome"] != r["outcome"] for k,r in normal.items()))
            print(name, label, summary[name][label], flush=True)
    save(HERE / "memory-probe.json", summary)


if __name__ == "__main__":
    main()
