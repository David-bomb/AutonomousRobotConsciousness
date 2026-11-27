from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
import torch

from maze.environment import Environment, Status
from ....data import level, save, read, digest
from ....teacher import Teacher
from .bank import HERE, CACHE, PACK, maps
from .features import MOVES, early_model, frame, phase, sign_index


def init():
    global net
    torch.set_num_threads(1)
    net = early_model(torch.load(PACK, weights_only=True)["early"])


def play(job):
    row, path = job
    if path.exists():
        with np.load(path) as f:
            return len(f["label"])
    env = Environment(level(row))
    teacher = Teacher(env.level)
    rng = np.random.default_rng(int(row["geometry"][:14], 16) ^ 20377)
    collected = []
    previous = None
    truth_policy = rng.random() < .5
    seen = set()
    for step in range(80):
        view = env.view
        if env.status != Status.RUNNING or phase(view.memory):
            break
        obs = frame(net, view)
        current = np.array([view.fuel_distance, view.repair_distance], np.int16)
        label = sign_index(previous, current)
        if label != 9:
            assert obs["allowed"][label], (row["name"], step, current, previous)
        truth = obs["missing"] if label == 9 else obs["options"][label]
        if step < 2:
            original = net(*legacy_batch(net, view, previous)).detach().numpy()[0]
            assert np.array_equal(truth.argmax(), original.argmax())
            ok = obs["legal"]
            np.testing.assert_allclose(truth[ok], original[ok] - original[ok].max(), atol=4e-5)
        allowed_q = obs["options"][obs["allowed"]]
        sensitive = len(set(allowed_q.argmax(1))) > 1 if len(allowed_q) else False
        key = view, label
        keep = key not in seen and (step < 5 or sensitive or (obs["legal"].sum() > 1 and rng.random() < .65))
        seen.add(key)
        if keep and len(collected) < 36:
            best, safe = teacher.labels(env)
            collected.append(dict(**obs, label=label, previous=np.full(2, -1, np.int16) if previous is None else previous,
                                  sensitive=sensitive, viable=safe, best=best, step=step, moved=bool(view.moved)))
        if len(collected) == 36:
            break
        action = int((truth if truth_policy else obs["missing"]).argmax())
        if rng.random() < .10:
            action = int(rng.choice(np.flatnonzero(obs["legal"])))
        previous = current
        env.step(MOVES[action])
    teacher.remaining.cache_clear()
    arrays = {k:np.stack([r[k] for r in collected]) for k in collected[0]}
    arrays["x"] = arrays["x"].astype(np.float16)
    with path.with_suffix(".pending").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    path.with_suffix(".pending").replace(path)
    return len(collected)


def legacy_batch(net, view, previous):
    from .features import legacy
    pair = (None, None) if previous is None else tuple(previous)
    return legacy.tensor_batch([net.features(view, pair)])


def main():
    cfg = dict(registration=digest(HERE/"registration.json"), source=digest(HERE/"collect.py"),
               features=digest(HERE/"features.py"))
    if (CACHE/"config.json").exists():
        assert read(CACHE/"config.json") == cfg
    else:
        save(CACHE/"config.json",cfg)
    with ProcessPoolExecutor(4, initializer=init) as pool:
        for split in ("fit", "held"):
            folder = CACHE / split
            folder.mkdir(exist_ok=True)
            bank = maps(split)
            pending = {pool.submit(play, (r, folder/(r["name"]+".npz"))):r for r in bank}
            counts = Counter()
            for n, future in enumerate(as_completed(pending), 1):
                counts[pending[future]["tier"]] += future.result()
                if n % 50 == 0:
                    print(split, n, counts, flush=True)
            save(HERE/(split+"-data.json"), dict(config=cfg, tiers=counts,
                 files={r["name"]:digest(folder/(r["name"]+".npz")) for r in bank}))


if __name__ == "__main__":
    main()
