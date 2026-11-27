"""Подбор параметров восьми выходов по итоговому SUCCESS новых полных стартов."""
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import torch
from ...data import OUT, digest, read, rows, save
from ...dev import evaluate
from ..belief.population import subset
from . import transfer
from .local_agent import LocalAgent, load
from .model import HERE


def package():
    step = read(HERE/"local-12000/results.json")["selected"]
    return dict(local=torch.load(HERE/f"local-12000/s{step}.pt", weights_only=True, map_location="cpu"),
                prefix=torch.from_numpy(np.fromfile(HERE/"prefix-counts.bin", np.int32).reshape(-1, 4)),
                foundation=torch.load(HERE.parent/"runs/joint400/rl.pt", weights_only=True, map_location="cpu"))


def init():
    torch.set_num_threads(1)
    transfer.net = LocalAgent(package())


def main():
    root = HERE.parent/"runs/local12000"
    root.mkdir(exist_ok=True)
    cache = HERE.parent/"cache/physical-local"
    cache.mkdir(exist_ok=True)
    zero = [1] + [0]*9
    options = [zero]
    for scale, old in ((4, 0), (9, -.5), (16, -1)):
        for recipe in ((1,1,0,0,0,0), (1,2,0,0,0,.5), (.5,1,.5,.5,.5,.5),
                       (0,1,0,1,0,.5), (0,0,0,1,1,1), (1,0,0,1,0,1)):
            options.append([1, old, *[scale*w for w in recipe], 2, 8])
    config = dict(options=options, screen="10 per tier", selection=200, rule="wins then minimum tier",
                  bank=digest(HERE/"reward-bank.json"), local=digest(HERE/"local-12000/results.json"),
                  prefix=digest(HERE/"prefix-counts.bin"), source=digest(HERE/"local_rl.py"),
                  controller=digest(HERE/"local_agent.py"))
    if (root/"config.json").exists():
        assert read(root/"config.json") == config
    else:
        save(root/"config.json", config)
    with ProcessPoolExecutor(3, initializer=init) as pool:
        screen = transfer.stage(pool, subset(rows(HERE/"reward-screen.jsonl"), 10), options, cache/"screen")
        save(root/"screen.json", screen)
        ranking = sorted(screen["rank"], key=lambda r: (-r["wins"], -r["minimum"]))
        finalists = [r["weights"] for r in ranking[:4]]
        if zero not in finalists:
            finalists.append(zero)
        val = transfer.stage(pool, rows(HERE/"reward-selection.jsonl"), finalists, cache/"selection")
        save(root/"selection.json", val)
    best = max(val["rank"], key=lambda r: (r["wins"], r["minimum"]))
    pack = package()
    pack["linear"] = torch.tensor(best["weights"], dtype=torch.float64)
    path = root/"rl.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save(pack, path)
    sig = digest(path)
    dev = OUT/"data/dev.jsonl"
    save(root/"locked-before-dev.json", dict(sha256=sig, selection=best, dev_sha256=digest(dev)))
    start = time.monotonic()
    summary, episodes = evaluate(load(path.read_bytes()), rows(dev))
    save(path.with_suffix(".physical.json"), dict(weights_sha256=sig, dev_sha256=digest(dev), summary=summary,
         episodes=episodes, seconds=time.monotonic()-start))
    print(best, summary, flush=True)


if __name__ == "__main__":
    main()
