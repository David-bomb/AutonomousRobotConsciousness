"""Select a frozen early policy by whole trips on two disjoint, tier-balanced training cohorts."""
import numpy as np
import torch

from early_trials import compare, physical
from exact_policy import ExactNet, load
from renewable_train import HERE, data, digest, save


def balanced(rows, per_tier):
    groups = [[r for r in rows if r["tier"] == tier][:per_tier] for tier in ("H1", "H2", "H3", "H4")]
    assert all(len(group) == per_tier for group in groups)
    return [row for quartet in zip(*groups, strict=True) for row in quartet]


def main():
    torch.set_num_threads(1)
    out = HERE / "exact-search"
    check = data.read(out / "true-check.json")
    assert check["after_first"] > 0 and min(r["weight"] for r in check["rows"]) > 0
    parent = HERE / "protected-search/fixed/model.pt"
    assert digest(parent) == "92b9b74fbd924161e79bf37b7e7b3dadfbafcd0bc2cd7fb0bbdcaee3d18665b1"
    bank = HERE / "handover-search/train.jsonl"
    maps = data.rows(bank)
    fit = balanced(maps[:100], 15)
    select = balanced(maps[100:], 25)
    assert not ({r["geometry"] for r in fit} & {r["geometry"] for r in select})
    eye = np.eye(12)
    candidates = np.vstack((np.zeros(12), eye[:10], 6*eye[10:], 12*eye[10:],
                            eye[6] + eye[7], .65*eye[2] + .65*eye[8]))
    pack = torch.load(parent, weights_only=True, map_location="cpu")
    pack.update(format="exact-mission-1", cap=1024, stop=18, exact=torch.zeros(12, dtype=torch.float64))
    save(out / "protocol.json", dict(parent_sha256=digest(parent), cap=pack["cap"], stop=pack["stop"],
         true_check_sha256=digest(out / "true-check.json"), bank_sha256=digest(bank),
         fit=[r["name"] for r in fit], select=[r["name"] for r in select], options=candidates.tolist(),
         sources={n:digest(HERE / n) for n in ("exact_train.py", "exact_policy.py", "exact_mission.dll",
                  "protected_policy.py", "origin_trace.py", "early_trials.py")},
         rule="Compare terminal SUCCESS on 15 training maps per tier. Advance four nonzero candidates with most finishes, then their mean and zero. On another 25 training maps per tier rank by total finishes, then H4 finishes, then prespecified order with parent first. Freeze and replay dev with normal two-register inputs and with both absent."))
    net = ExactNet(pack)
    first = compare(out, "fit", fit, candidates, net)
    leaders = candidates[np.argsort(-first[1:],kind="stable")[:4]+1]
    pool = np.vstack((candidates[0], leaders, leaders.mean(0)))
    second = compare(out, "select", select, pool, net)
    episodes = data.read(out / "select.json")["episodes"]
    h4 = [sum(e["tier"] == "H4" and e["model"] == j and e["outcome"] == "SUCCESS" for e in episodes)
          for j in range(len(pool))]
    winner = max(range(len(pool)), key=lambda j: (second[j], h4[j], -j))
    dest = out / "model.pt"
    if not dest.exists():
        tmp = dest.with_suffix(".tmp")
        torch.save(dict(pack, exact=torch.tensor(pool[winner], dtype=torch.float64)), tmp)
        tmp.replace(dest)
    save(out / "frozen.json", dict(model_sha256=digest(dest), fit=first.tolist(), select=second.tolist(),
         h4=h4, winner=winner, coefficients=pool[winner].tolist(), protocol_sha256=digest(out / "protocol.json")))
    for tag, ablate in (("dev", False), ("ablated", True)):
        physical(out, tag, load(dest.read_bytes(), ablate))


if __name__ == "__main__":
    main()
