"""Проверить состав проб и оба журнала действий, заново собрать сводку."""
from collections import Counter
from json import dumps, loads
from pathlib import Path
from statistics import median
import sys

from run import (HERE, MODES, MOVES, THRESHOLD, TIERS, Environment, Status, digest,
                 group, level, loss, read, registration, state, train, write)


def verify(out):
    lock_sha = digest((HERE/"lock.json").read_bytes())
    timing = read(out/"timing-rule.json")
    assert lock_sha == "6f391b4f68d1995238e72ae99b0fce622fb7dfb51a97254014d375a20feb2efc"
    assert timing["lock_sha256"] == lock_sha and read(HERE/"lock.json")["registration"] == registration()
    rows = train()
    n = 4 if timing["probe_seconds"] > THRESHOLD else 8
    assert timing["per_tier"] == n and timing["pilot"] == group(rows,1)
    selected = group(rows,n)
    paths = {f"{i}-{m}.json" for i in selected for m in MODES}
    assert paths == {p.name for p in (out/"episodes").iterdir()}
    paired = {}
    results = {m:[] for m in MODES}
    hashes = {}
    for i in selected:
        row = rows[i]
        pair = {}
        for mode in MODES:
            p = out/"episodes"/f"{i}-{mode}.json"
            r = read(p)
            assert (r["id"],r["tier"],r["geometry"],r["initial"],r["index"],r["mode"],r["split"],r["family"]) == (
                row["name"],row["tier"],row["geometry"],[row["fuel"],row["health"]],i,mode,"sequence/train",row["family"])
            assert r["scenario_sha256"] == digest(dumps([row["grid"],row["fuel"],row["health"]],
                                                         separators=(",",":")).encode())
            env = Environment(level(row))
            assert env.limit == r["limit"]
            numbers = bytes(int(c) for c in r["actions"])
            assert digest(numbers) == r["actions_sha256"] and len(numbers) == r["steps"]
            states, actions = [state(env)], []
            for number in numbers:
                move = MOVES[number]
                env.step(move)
                actions.append(move)
                states.append(state(env))
            assert env.status != Status.RUNNING
            assert (env.status.name,env.steps,env.refuels,env.repairs) == tuple(r[k] for k in
                                                                                ("outcome","steps","refuels","repairs"))
            if env.status == Status.SUCCESS:
                assert r["first_loss"] is None
            else:
                assert r["first_loss"] == loads(dumps(loss(env.level,env.limit,states,actions)))
            results[mode].append(r)
            pair[mode] = r
            hashes[p.name] = digest(p.read_bytes())
        full, prev = (pair[m] for m in MODES)
        first_diff = next((t for t,(a,b) in enumerate(zip(full["actions"],prev["actions"]),1) if a != b), None)
        if first_diff is None:
            assert full["actions"] == prev["actions"]
        paired[i] = dict(id=row["name"],tier=row["tier"],first_action_difference=first_diff,
                         **{m:{k:r[k] for k in ("outcome","steps","refuels","repairs","elapsed_seconds")} |
                            {"first_loss_step": r["first_loss"]["step"] if r["first_loss"] else None} for m,r in pair.items()})
    stats = {}
    for m, records in results.items():
        failed = [r["first_loss"] for r in records if r["first_loss"]]
        stats[m] = dict(outcomes=dict(Counter(r["outcome"] for r in records)),
                        steps=sum(r["steps"] for r in records), elapsed_sum_seconds=sum(r["elapsed_seconds"] for r in records),
                        tier_successes={t:sum(r["tier"]==t and r["outcome"]=="SUCCESS" for r in records) for t in TIERS},
                        loss_phases=dict(Counter(f["phase"] for f in failed)),
                        median_first_loss_step=median(f["step"] for f in failed) if failed else None)
    wins = Counter()
    for pair in paired.values():
        a,b = (pair[m]["outcome"] == "SUCCESS" for m in MODES)
        wins["both" if a and b else "full_only" if a else "previous_only" if b else "neither"] += 1
    return dict(lock_sha256=lock_sha, timing=timing, wall=read(out/"wall.json"), per_tier=n,
                maps=len(selected), episodes=len(hashes), stats=stats, paired_successes=wins,
                pairs=paired, episode_sha256=hashes)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv)>1 else HERE/"physical"
    result = loads(dumps(verify(out)))
    path = out/"receipt.json"
    if path.exists():
        assert read(path) == result
    else:
        write(path, result)
    print(dumps({k:result[k] for k in ("maps","episodes","stats","paired_successes","wall")}, ensure_ascii=False))
