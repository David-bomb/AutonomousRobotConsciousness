"""Внешний предел сочетания стратегий по уже выполненным полным сериям."""
from hashlib import sha256
from itertools import combinations
from collections import Counter
import json

from bench import HERE, SETS
from ledger import PARENTS


def diagnose_uncovered(snapshot, unsolved):
    """Объясняет отказ street-trunk на картах, где ни у кого нет записанного финиша."""
    def indexed(filename):
        return {row["map"]:row for line in (HERE/filename).read_bytes().splitlines() if (row:=json.loads(line))}

    decisions = indexed("target-or-direction.jsonl")
    windows = indexed("next-observation.jsonl")
    maps = {}
    totals = {}
    for split, names in unsolved.items():
        counts = Counter()
        for name in names:
            row = decisions[name]
            if name in windows:
                actual = windows[name]["actual"]
                if not actual:
                    reason = "поздний выезд без нового наблюдения"
                elif actual["can_finish"]:
                    reason = "финиш утрачен после первого нового окна"
                elif actual["possible_at_same_exit"]:
                    reason = "не хватило запасов при первом новом окне"
                else:
                    reason = "первое новое окно исключало финиш"
            elif row["viable_downhill"]:
                reason = "другое направление к выбранному сервису сохраняло финиш"
            else:
                reason = "сокращение пути к выбранному сервису не сохраняло финиш"
            counts[reason] += 1
            maps[name] = dict(reason=reason,split=split,family=row["family"],step=row["critical"]["step"])
        totals[split] = dict(counts)
    (HERE/"uncovered-diagnosis.json").write_text(json.dumps(
        dict(source=snapshot.name,controller="street-trunk",counts=totals,maps=maps),
        ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def main():
    keys = None
    masks = {}
    digests = {}
    for name in ("reserve", *PARENTS):
        path = HERE/f"{name}.jsonl"
        if not path.exists():
            continue
        data = path.read_bytes()
        lines = data.splitlines()
        if len(lines) != 336:
            continue
        runs = [json.loads(line) for line in lines]
        rows = {(r["split"],r["map"]):r for r in runs}
        if keys is None:
            keys = sorted(rows)
        assert sorted(rows) == keys and len(rows) == 336
        masks[name] = sum(1<<i for i,k in enumerate(keys) if rows[k]["status"]=="SUCCESS")
        digests[name] = sha256(data.replace(b"\r\n", b"\n")).hexdigest()

    groups = [sum(1<<i for i,k in enumerate(keys) if k[0]==split) for split in SETS]
    assert [g.bit_count() for g in groups] == [100,100,100,36]
    def counts(mask):
        return [(mask & group).bit_count() for group in groups]

    everyone = 0
    for mask in masks.values():
        everyone |= mask
    summaries = []
    for size in (1,2,3):
        best = None
        considered = qualifying = 0
        for members in combinations(masks,size):
            together = 0
            for name in members:
                together |= masks[name]
            wins = counts(together)
            if wins[3] != 36:
                continue
            considered += 1
            floor = min(wins[:3])
            qualifying += floor >= 85
            rank = floor, sum(wins[:3])
            if best is None or rank > best[0]:
                best = rank, members, wins
        _,members,wins = best
        summaries.append(dict(size=size,full_val=considered,at_target=qualifying,
                              example=members,upper=wins))
    unsolved = {split:[name for i,(s,name) in enumerate(keys) if s==split and not everyone & (1<<i)]
                for split in SETS}
    result = dict(series=len(masks), source_sha256_lf=digests, all_upper=counts(everyone),
                  unsolved=unsolved, combinations=summaries)
    tag=sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()[:12]
    snapshot=HERE/f"coverage-{len(masks)}-{tag}.json"
    contents=json.dumps(result,indent=2)+"\n"
    if snapshot.exists():
        assert json.loads(snapshot.read_text(encoding="utf8"))==json.loads(contents)
    else:
        snapshot.write_text(contents,encoding="utf8")
    (HERE/"coverage-bounds.json").write_text(contents,encoding="utf8")
    if (HERE/"next-observation.jsonl").exists():
        diagnose_uncovered(snapshot,unsolved)
    print(result["series"],result["all_upper"],summaries,flush=True)


if __name__=="__main__":
    main()
