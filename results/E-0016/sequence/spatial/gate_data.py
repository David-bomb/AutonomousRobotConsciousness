"""Разметка выходов истинной дорожной сетью. Тот же извлекатель при оценке видит только память."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from maze.grid import wave, locations
from ...data import digest, read, rows, save
from .gates import HERE, describe


def site_key(grid):
    return tuple(y*40+x for symbol in "FRB" for x,y in locations(grid, symbol))


def job(args):
    bank, source, output = args
    if output.exists():
        return
    grids = {site_key(tuple(row["grid"])): tuple(row["grid"]) for row in bank}

    @lru_cache(maxsize=100)
    def lengths(key, target):
        distances = np.full(1600, 300, np.int16)
        pos = (target % 40, target // 40)
        for (x,y), length, _ in wave(grids[key], [pos]):
            distances[y*40+x] = length
        return distances

    records = []
    with np.load(source) as data:
        boards, metas, ys, viable, best, stocks = [data[n] for n in ("maps","meta","y","viable","best","stocks")]
    for board, meta, signs, safe, fast, stock in zip(boards, metas, ys, viable, best, stocks):
        x, ref, urgent = describe(board, meta, stock)
        legal = signs[:,0] >= 0
        down = signs[:,urgent] == 0
        assert np.all((x[...,0].sum(1)==0) | legal)
        key = tuple(meta[1:6])
        good = np.zeros((8,8), bool)
        mask = ref[:,:,0] >= 0
        p, q, steps, end = ref[mask].T
        topology = lengths(key, key[-1]) < 300
        available = topology[q] & topology[(p//40)*40 + q%40] & topology[(q//40)*40 + p%40]
        for target in np.unique(end):
            take = end == target
            available[take] &= steps[take] + lengths(key, target)[q[take]] == meta[6+urgent]
        good[mask] = available
        assert not np.any(good.any(1) & ~down)
        records.append((x.astype(np.float16), good, legal, down, safe, fast))
    with output.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **{k:np.stack(v) for k,v in zip(("x","witness","legal","down","safe","fast"),zip(*records))})
    output.with_suffix(".tmp").replace(output)


def main():
    bank = rows(HERE/"extra.jsonl")
    previous = read(HERE/"early.json")
    cache = HERE.parent/"cache/gates"
    cache.mkdir(exist_ok=True)
    manifest = []
    for group, bank_part in zip(previous["sets"], (bank[:3000], bank[3000:])):
        jobs = []
        for i,name in enumerate(group["parts"]):
            jobs.append((bank_part[i*20:i*20+20], HERE.parent/"cache/spatial"/name, cache/name))
        with ProcessPoolExecutor(2) as pool:
            for j,_ in enumerate(pool.map(job, jobs),1):
                if j%25==0:
                    print(group["split"], j*20, flush=True)
        positives=covered=states=0
        for _,_,path in jobs:
            with np.load(path) as data:
                positives+=int(data["down"].sum())
                covered+=int(data["witness"].any(2).sum())
                states+=len(data["down"])
        manifest.append(dict(split=group["split"],states=states, actual_downhill=positives,
                    represented_downhill=covered, parts={p.name:digest(p) for _,_,p in jobs}))
    save(HERE/"gate-data.json", dict(sets=manifest, input=digest(HERE/"early.json"), source=digest(HERE/"gate_data.py"),
          representation=digest(HERE/"gates.cpp")))


if __name__ == "__main__":
    main()
