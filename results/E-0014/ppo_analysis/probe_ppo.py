"""Чувствительность вероятностей PPO в парных ресурсных и пространственных развилках."""
from collections import Counter
import gzip
import json
import statistics

import numpy as np
import torch

from check_ppo import OUT, ROOT
from neural_rl.corpus_study.diagnostic_eval import expected_moves
from neural_rl.corpus_study.diagnostics import probe_view
from neural_rl.corpus_study.learn import setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import encode


@torch.inference_mode()
def compare(net, group, rows):
    pairs = []
    assert len(rows) == 50
    for a, b in zip(rows[::2], rows[1::2]):
        assert a["pair"] == b["pair"]
        views = [probe_view(r) for r in (a, b)]
        states = [encode(v, False) for v in views]
        s1, s2 = states
        assert np.array_equal(s1.legal, s2.legal)
        diff = np.flatnonzero(s1.numbers != s2.numbers).tolist()
        if group == 4:
            assert not diff and not np.array_equal(s1.image, s2.image)
        elif group == 6:
            assert diff in ([4], [5]) and np.array_equal(s1.image, s2.image)
        else:
            assert np.array_equal(s1.image, s2.image) and not diff
        expected = [expected_moves(r, v) for r, v in zip((a, b), views)]
        assert all(len(moves) == 1 for moves in expected) and expected[0] != expected[1]
        probs = net(batch(states))[0].softmax(-1).cpu().numpy()
        choices = probs.argmax(axis=1).tolist()
        pairs.append(dict(pair=a["pair"], maps=[a["name"], b["name"]], choices=choices,
                          expected=[m[0] for m in expected], probabilities=probs.tolist(),
                          moved_probability=float(abs(probs[0] - probs[1]).sum()/2)))
    return dict(pairs=pairs, median_probability_shift=statistics.median(p["moved_probability"] for p in pairs),
                max_probability_shift=max(p["moved_probability"] for p in pairs),
                changed=sum(p["choices"][0] != p["choices"][1] for p in pairs),
                correct=sum(sum(c == e for c, e in zip(p["choices"], p["expected"])) for p in pairs))


def main():
    original = read(ROOT / "results/E-0012/frozen.json")["ppo"]
    folder = ROOT / "results/E-0012/runs" / original["id"]
    diag = ROOT / "results/E-0012/diagnostics"
    control_path = diag / "control.jsonl.gz"
    assert digest(control_path) == read(diag / "verified.json")["sha256"]["control"]
    with gzip.open(control_path, "rt", encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    config = read(folder / "run.json")["config"]
    setup(config["seed"])
    net = Network(config).cuda().eval()
    result = {}
    copied = diag / "training/ppo"
    for version, path, fingerprint in [("main", folder / "weights.pt", original["sha256"]),
                                       ("diag", copied / "weights.pt", read(copied / "run.json")["sha256"])]:
        assert digest(path) == fingerprint
        net.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        result[version] = {g: compare(net, g, [r for r in rows if r["group"] == g]) for g in (4, 6, 11)}
        for g, stats in result[version].items():
            print(version, g, {k:v for k,v in stats.items() if k != "pairs"})
    before = read(diag / "score-ppo.json")["groups"]
    after = read(copied / "control.json")["rows"]
    wins = Counter(r["stage"] for r in after if r["outcome"] == "SUCCESS")
    result["diagnostic_finishes_by_group"] = {g: [d["wins"], wins[int(g)]] for g, d in before.items()}
    save(OUT / "probes.json", result)


if __name__ == "__main__":
    main()
