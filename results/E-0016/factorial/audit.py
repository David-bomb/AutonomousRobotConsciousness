"""Проверка парности, состава колод, независимости геометрий и точности пяти повторов."""
import gzip
import json
import math
import os
import random

for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

from .analyze import ARMS, OUT, digest, need, read, save


def first_deck(names, seed):
    deck = list(names)
    random.Random(seed).shuffle(deck)
    return deck[:24]


def sensitivity():
    # T=(Z+d*sqrt(5))/sqrt(V/4), V имеет распределение χ²(4).
    # Интегрируем по x=V/2 с плотностью x*exp(-x).
    x, w = np.polynomial.laguerre.laggauss(80)
    crit = 2.776445105
    def power(effect):
        shift = effect*math.sqrt(5)
        terms = []
        for sample in x:
            cut = crit*math.sqrt(sample/2)
            terms.append((math.erfc((cut-shift)/math.sqrt(2)) +
                          math.erfc((cut+shift)/math.sqrt(2)))/2)
        return float(np.sum(w*x*terms))
    lo, hi = 0., 4.
    for _ in range(50):
        mid = (lo+hi)/2
        if power(mid) < .8:
            lo = mid
        else:
            hi = mid
    assert abs(power(0)-.05) < .0001 and abs(power(hi)-.8) < 1e-10
    return dict(paired_t_model_n=5, two_sided_alpha=.05, target_power=.8,
                minimum_effect_in_units_of_seed_difference_sd=hi,
                minimum_two_sided_exact_sign_p=2/2**5)


def main():
    reg = read(OUT/"registration.json")
    index = json.loads(gzip.decompress((OUT/"scenarios.json.gz").read_bytes()))
    train = sorted(k for k in index if k.startswith("train-"))
    dev = sorted(k for k in index if k.startswith("validation_id-"))
    train_hashes = {index[k]["geometry"] for k in train}
    dev_hashes = {index[k]["geometry"] for k in dev}
    assert len(train_hashes) == 30000 and len(dev_hashes) == 60 and not train_hashes & dev_hashes
    ordered = sorted(train,key=lambda k:(index[k]["deficits"],index[k]["direct"],k))
    c1 = []
    for stage in (1,2,3):
        group = ordered[(stage-1)*40000:stage*40000]
        assert all(index[k]["c1"] == stage for k in group)
        c1.append(dict(stage=stage, scenarios=len(group), by_deficit={
            label:sum(need(index[k])==label for k in group) for label in ("0","F","R","FR")}))
    c0_first = [k for k in train if index[k]["stage"] == 1]
    c1_first = ordered[:40000]
    checks = []
    for phase in ("technical","full"):
        result = read(OUT/phase/"summary.json")
        ticks = 256 if phase=="technical" else 1024
        for seed in result["seeds"]:
            for arm in ARMS:
                folder = OUT/phase/str(seed)/arm
                record = read(folder/"training.json")
                proof = read(folder/"verified.json")
                assert proof["registration"] == digest(OUT/"registration.json")
                batch = list(record["history"][0]["maps"])[:24]
                assert batch == first_deck(c1_first if arm.endswith("C1") else c0_first, seed)
                assert all(h["steps"] == 24*ticks and h["updates"] == ticks//2 for h in record["history"])
                assert all(k.startswith("train-") and k in index for h in record["history"] for k in h["maps"])
                assert record["initial"] == reg["initial"][str(seed)] != record["final"]
                assert all(digest(folder/k) == v for k,v in proof["files"].items())
                rows = read(folder/"validation.json")
                assert [r["map"] for r in rows] == dev
                checks.append(dict(phase=phase, seed=seed, arm=arm, train_seconds=record["train_seconds"],
                                   eval_seconds=proof["evaluation_seconds"], initial=record["initial"],
                                   final=record["final"], transitions=72*ticks, updates=3*ticks//2,
                                   physical_validation_steps=sum(r["steps"] for r in rows),
                                   max_R1_identity_error=max(r["residual_R1"] for r in rows),
                                   matches_reference=len(rows), verified_file_sha256=digest(folder/"verified.json")))
    assert len(set(v["initial"] for v in checks)) == 5
    assert len(set(v["final"] for v in checks)) == 24
    save(OUT/"audit.json", dict(registration=digest(OUT/"registration.json"), train_dev_overlap=0,
                                c1=c1, first_24_draws_match_all_runs=True, checks=checks,
                                design_sensitivity=sensitivity(),
                                analysis_code={f:digest(OUT/f) for f in ("analyze.py","audit.py")}))
    print("Verified", len(checks), "models and", sum(v["matches_reference"] for v in checks), "episodes")


if __name__ == "__main__":
    main()
