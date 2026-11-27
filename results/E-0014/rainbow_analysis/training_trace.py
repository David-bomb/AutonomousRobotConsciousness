"""Наблюдение за повтором закреплённого обучения Rainbow без изменения алгоритма."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from neural_rl import training
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network
from neural_rl.rollout import evaluate


def score(net, pool):
    return [r for i in range(0, len(pool.cases), 32)
            for r in evaluate(net, Chunk(pool, pool.cases[i:i + 32]), 1)]


def ending(item):
    return ("win" if item[2] > 0 else "loss") if item[3] == 0 else "ongoing"


def main(plain_repeat=False):
    source = HERE.parents[1] / "E-0012"
    choice = read(source / "frozen.json")["rainbow"]
    folder = source / "runs" / choice["id"]
    assert digest(folder / "weights.pt") == choice["sha256"]
    old = read(folder / "run.json")
    cfg = old["config"]
    setup(cfg["seed"])
    train, val = Pool("train", cfg["geometry_count"]), Pool("validation_id", 60)
    net = Network(cfg).cuda()
    before = score(net, val)
    expected = read(folder / "validation.json")
    assert before == expected["before"]
    net.train()
    rng = random.Random(cfg["seed"])
    counts = Counter()
    sample_counts = Counter()
    uniform_expected = Counter()
    contents = Counter()
    terminal = Counter()
    project = training.project
    advance, forward = training.Stage.advance, Network.forward
    sample, add = training.Replay.sample, training.Replay.add

    def observe_step(stage, lane, *args):
        env = stage.environments[lane]
        result = advance(stage, lane, *args)
        if env.status.name != "RUNNING":
            _, index, scenario = env.level.title.split("-")
            direct = train.records(int(index))[int(scenario)]["direct"]
            required = ("both" if max(env.level.fuel, env.level.health) < direct
                        else "one" if min(env.level.fuel, env.level.health) < direct else "neither")
            terminal[required, env.status.name] += 1
        return result

    def observe_forward(network, data):
        out = forward(network, data)
        if network is net and network.training and not torch.is_grad_enabled() and len(data[0]) == cfg["lanes"]:
            network.eval()
            quiet = forward(network, data)[0].argmax(1)
            network.train()
            counts["collection_choices"] += len(quiet)
            counts["noise_changed_argmax"] += int((quiet != out[0].argmax(1)).sum())
        return out

    def observe_add(replay, item):
        if len(replay.items) == len(replay.priority):
            contents[ending(replay.items[replay.cursor])] -= 1
        contents[ending(item)] += 1
        return add(replay, item)

    def observe_sample(replay, count, rainbow, rng, fraction):
        result = sample(replay, count, rainbow, rng, fraction)
        sample_counts.update(ending(replay.items[i]) for i in result[0])
        n = len(replay.items)
        uniform_expected.update({k: count * v / n for k, v in contents.items()})
        return result

    def observe_project(probs, rewards, discounts, support):
        z = rewards[:, None] + discounts[:, None] * support
        proj = project(probs, rewards, discounts, support)
        mass_outside = (probs * ((z < support[0]) | (z > support[-1]))).sum()
        counts["target_distributions"] += len(z)
        counts["sum_mass_outside_support"] += float(mass_outside)
        counts["max_projection_mass_error"] = max(counts["max_projection_mass_error"],
                                                   float((proj.sum(1) - 1).abs().max()))
        return proj

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    with ExitStack() as stack:
        if not plain_repeat:
            for obj, method, observer in ((training.Stage, "advance", observe_step),
                                          (Network, "forward", observe_forward),
                                          (training.Replay, "add", observe_add),
                                          (training.Replay, "sample", observe_sample),
                                          (training, "project", observe_project)):
                stack.enter_context(patch.object(obj, method, observer))
        history = training.train_q(net, optimizer, cfg, train, rng, np.random.default_rng(cfg["seed"]))

    rows = score(net, val)
    archived = torch.load(folder / "weights.pt", weights_only=True, map_location="cuda")
    delta = max(float((state.float() - archived[name].float()).abs().max())
                for name, state in net.state_dict().items())
    terminal_rows = {group: {outcome: n for (g, outcome), n in terminal.items() if g == group}
                     for group in ("neither", "one", "both")}
    result = dict(id=choice["id"], archived_sha256=choice["sha256"],
                  before_rows_equal=True, max_absolute_weight_difference=delta,
                  matching_after_rows=sum(a == b for a, b in zip(rows, expected["after"])),
                  archive_after_wins=old["screen"]["wins"], repeat_after_wins=sum(r["outcome"] == "SUCCESS" for r in rows),
                  hardware=torch.cuda.get_device_name(), train_steps=sum(s["steps"] for s in history),
                  stage_outcomes=[s["outcomes"] for s in history],
                  stage_outcomes_archived=[s["outcomes"] for s in old["training"]],
                  terminal_groups=terminal_rows, sampled_nstep_endings=dict(sample_counts),
                  expected_nstep_endings_uniform=dict(uniform_expected), instrumentation=dict(counts))
    if not plain_repeat:
        assert counts["collection_choices"] == result["train_steps"] and sum(sample_counts.values()) == counts["target_distributions"]
    save(HERE / ("plain-repeat.json" if plain_repeat else "training-trace.json"), result)
    print(result)


if __name__ == "__main__":
    main(plain_repeat="--plain" in sys.argv)
