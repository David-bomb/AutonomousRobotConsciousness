"""Парные полные финиши одинаковых стартов после фиксации весов."""
from collections import Counter

from ....data import OUT, digest, read, rows, save
from .cells import HERE


def summarize(pairs):
    counts = Counter()
    for base, room, _ in pairs:
        outcomes = (base["outcome"] == "SUCCESS", room["outcome"] == "SUCCESS")
        counts[outcomes] += 1
    return dict(
        starts=len(pairs), base=counts[True, False] + counts[True, True],
        room=counts[False, True] + counts[True, True],
        gained=counts[False, True], lost=counts[True, False],
        both_services_base=sum(bool(a["refuels"] and a["repairs"]) for a, _, _ in pairs),
        both_services_room=sum(bool(b["refuels"] and b["repairs"]) for _, b, _ in pairs),
    )


def compare(pairs):
    by_h4 = {}
    for family in ("corridors", "branching", "loops", "rooms"):
        subset = [p for p in pairs if p[2]["tier"] == "H4" and p[2]["family"] == family]
        by_h4[family] = summarize(subset)
    return dict(total=summarize(pairs), h4=by_h4)


def choices(episodes):
    result = {}
    for key in ("first_tight", "first_loss"):
        result[key] = {}
        for family in ("all", "rooms", "loops"):
            for tier in ("all", "H4"):
                tests = [r["tests"][key] for r in episodes if key in r["tests"]
                         and (family == "all" or r["family"] == family)
                         and (tier == "all" or r["tier"] == tier)]
                result[key][family + "_" + tier] = dict(
                    count=len(tests), changed=sum(t["changed"] for t in tests),
                    gained=sum(t["candidate_safe"] and not t["previous_safe"] for t in tests),
                    lost=sum(t["previous_safe"] and not t["candidate_safe"] for t in tests),
                    classified_as_rooms=sum(t["family_pred"] == 3 for t in tests),
                    core_roads=sum(t["inferred"] for t in tests), real_roads=sum(t["actually_road"] for t in tests),
                )
    return result


def main():
    runs = HERE.parent.parent / "runs"
    folder = runs / "rooms3000"
    lock = read(folder / "locked-before-dev.json")
    assert digest(folder / "rl.pt") == lock["weights_sha256"]
    dev = read(folder / "rl.physical.json")
    original = read(runs / "field3000-clean/rl.physical.json")
    assert dev["dev_sha256"] == original["dev_sha256"] == digest(OUT / "data/dev.jsonl")
    assert dev["weights_sha256"] == lock["weights_sha256"]
    dev_pairs = list(zip(original["episodes"], dev["episodes"], rows(OUT / "data/dev.jsonl"), strict=True))
    assert all(a["map"] == b["map"] == row["name"] for a, b, row in dev_pairs)

    selection = read(folder / "selection.json")
    weights = [item["weights"] for item in selection["rank"]]
    base, room, pre = [weights.index(lock[k]["weights"]) for k in ("baseline", "room_candidate", "pre_RL")]
    bank = {row["name"]: row for row in rows(HERE / "selection.jsonl")}
    def pairs(i, j):
        return [(r["episodes"][i], r["episodes"][j], bank[r["map"]]) for r in selection["records"]]

    independent = read(folder / "independent.json")
    assert independent["config"]["weights"] == lock["weights_sha256"]
    independent_pairs = [(r["runs"]["base"], r["runs"]["room"], r) for r in independent["episodes"]]
    pre_result = read(folder / "independent-pre.json")
    assert pre_result["config"]["bank"] == independent["config"]["data"]
    assert pre_result["config"]["weights"] == digest(folder / "pre.pt")
    pre_pairs = [(p, r["runs"]["room"], r) for p, r in zip(pre_result["episodes"], independent["episodes"], strict=True)]
    assert all(a["map"] == b["map"] for a, b, _ in pre_pairs)
    output = dict(
        weights_sha256=lock["weights_sha256"],
        baseline_sha256=original["weights_sha256"], selection=compare(pairs(base, room)),
        selection_before_after_RL=compare(pairs(pre, room)), dev=compare(dev_pairs),
        independent=compare(independent_pairs), independent_before_after_RL=compare(pre_pairs),
        independent_early_choices=choices(independent["episodes"]),
        sources={n: digest(folder / n) for n in ("selection.json", "locked-before-dev.json",
                                                "rl.physical.json", "independent.json", "independent-pre.json")},
    )
    save(folder / "comparison.json", output)
    print({k: v["total"] for k, v in output.items() if isinstance(v, dict) and "total" in v})


if __name__ == "__main__":
    main()
