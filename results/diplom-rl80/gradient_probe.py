"""Fit aggregate train path prefixes and compare held-out sensor-descent predictions."""
from collections import Counter
import ctypes
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3] / "University/Diplom"
sys.path.insert(0, str(PROJECT))
from maze.environment import Environment, Status
from maze.grid import destination

data = import_module("results.E-0016.data")
policy = import_module("results.E-0016.policy")
teacher = import_module("results.E-0016.teacher")

native = ctypes.CDLL(str(HERE / "gradient_prefix.dll"))
cint, cstr = ctypes.c_int, ctypes.c_char_p
native.fit.argtypes = [cstr, cint, cint, ctypes.c_uint]
native.fit.restype = None
native.predict.argtypes = [cstr, cint, cint, cint, cint, cint, np.ctypeslib.ndpointer(np.float64)]
native.predict.restype = None
native.save_bank.argtypes = native.load_bank.argtypes = [cstr]
native.save_bank.restype = native.load_bank.restype = cint


def predict(view, kind, distance):
    mem = view.memory
    h, w = len(mem), len(mem[0])
    x, y = view.position
    counts = np.zeros(8, dtype=np.float64)
    native.predict("".join(mem).encode("ascii"), w, h, y*w+x, distance, ord(kind), counts)
    return counts


def inspect(row, net):
    env = Environment(data.level(row))
    oracle = teacher.Teacher(env.level)
    records = []
    seen = set()
    while env.status == Status.RUNNING and env.view not in seen:
        v = env.view
        seen.add(v)
        x, legal = policy.features(v)
        raw = net(torch.from_numpy(x[None]), torch.from_numpy(legal[None]))[0].numpy()
        pilot = int(raw.argmax())
        readings = [(v.fuel-v.fuel_distance, "F", v.fuel_distance, 0),
                    (v.health-v.repair_distance, "R", v.repair_distance, 2)]
        slack, kind, distance, offset = min(readings)
        start = env.steps == 0
        near = legal.sum() > 1 and slack <= 2 and 0 < distance
        if start or near:
            counts = predict(v, kind, distance)
            counts[~legal] = -1
            candidate = max(np.flatnonzero(legal).tolist(), key=lambda i: (counts[i], raw[i]))
            closer = []
            for i in np.flatnonzero(legal):
                q = destination(v.memory, v.position, policy.MOVES[i])
                if min(field[q] for field in oracle.dist[offset:offset+2]) < distance:
                    closer.append(int(i))
            _, viable = oracle.labels(env)
            records.append(dict(geometry=row["geometry"], tier=row["tier"], step=env.steps,
                                start=start, near=bool(near), slack=slack, distance=distance,
                                covered=counts.max()>0, chosen=candidate, pilot=pilot, closer=closer,
                                viable=np.flatnonzero(viable).tolist()))
        if near:
            break
        env.step(policy.MOVES[pilot])
    return records


def main():
    torch.set_num_threads(1)
    exp = PROJECT / "results/E-0016"
    bank_path = HERE / "gradient-prefix.bin"
    train_path = exp / "data/train.jsonl"
    rows = data.rows(train_path)
    assert len(rows)==800
    counts = []
    for i, item in enumerate(rows[:600]):
        grid = item["grid"]
        native.fit("".join(grid).encode("ascii"), len(grid[0]), len(grid), 27003+i)
    temp = bank_path.with_suffix(".bin.tmp")
    n = native.save_bank(os.fsencode(temp))
    if n <= 0: raise OSError("Could not save aggregate path counts")
    if bank_path.exists() and bank_path.read_bytes()!=temp.read_bytes():
        raise ValueError("Existing bank differs")
    temp.replace(bank_path)
    assert native.load_bank(os.fsencode(bank_path)) == n
    model = policy.Policy().eval()
    weights = exp / "runs/pilot200/rl.pt"
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        for item in rows[600:]:
            counts.extend(inspect(item, model))
    groups = {}
    for key in ("start", "near"):
        subset=[r for r in counts if r[key]]
        groups[key] = dict(n=len(subset), covered=sum(bool(r["covered"]) for r in subset))
        for what in ("closer","viable"):
            totals=Counter((r["pilot"] in r[what],r["chosen"] in r[what]) for r in subset)
            groups[key][what] = dict(pilot=totals[True,True]+totals[True,False],
                                     prefix=totals[True,True]+totals[False,True],
                                     gained=totals[False,True], lost=totals[True,False])
    output=dict(groups=groups, n_prefix_bins=n, bank_sha256=sha256(bank_path.read_bytes()).hexdigest(),
                train_sha256=sha256(train_path.read_bytes()).hexdigest(), weights_sha256=sha256(weights.read_bytes()).hexdigest(),
                fit="first 600 of E-0016 train", measured="last 200 of E-0016 train", records=counts)
    out = HERE / "gradient-probe.json"
    encoded=(json.dumps(output,ensure_ascii=False,indent=2,default=lambda x:x.item())+"\n").encode()
    if out.exists() and out.read_bytes()!=encoded: raise ValueError("Previous measurement differs")
    out.write_bytes(encoded)
    print(json.dumps(groups))


if __name__ == "__main__":
    main()
