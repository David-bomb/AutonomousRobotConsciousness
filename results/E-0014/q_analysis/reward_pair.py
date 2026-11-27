"""Пять пар Double DQN v03 с изменением одного слагаемого награды. Приёмка на validation_id."""
import argparse
from collections import Counter, defaultdict
import json
import random

import numpy as np
import torch

from analyze import HERE, SRC, junctions
from maze.datasets import Case
from neural_rl import rollout
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read
from neural_rl.networks import Network, batch
from neural_rl.training import train_q
from neural_rl.rollout import evaluate

ORIGINAL = rollout.reward
OUT = HERE / "reward_pair"


def without_coordinates(before, after, status):
    geo = max(abs(before.dx),abs(before.dy)) - max(abs(after.dx),abs(after.dy))
    return ORIGINAL(before,after,status) - 0.35*geo


def train(config, reward):
    setup(config["seed"])
    rng = random.Random(config["seed"])
    pool = Pool("train",config["geometry_count"])
    net = Network(config).cuda().train()
    opt = torch.optim.Adam(net.parameters(),lr=config["lr"])
    rollout.reward = reward
    try:
        history = train_q(net,opt,config,pool,rng,np.random.default_rng(config["seed"]))
    finally:
        rollout.reward = ORIGINAL
    assert [s["steps"] for s in history] == [24576]*3
    assert [s["updates"] for s in history] == [512]*3
    return net.eval(), [s["outcomes"] for s in history]


def probes(validation):
    out = []
    for entry in validation.cases:
        env = validation.start(entry)
        record = validation.records(entry.index)[entry.scenario]
        for _, group, state, _, optimal, _, _ in junctions(Case(entry.name,entry.stage,env.level), {"actions":record["moves"]}):
            out.append((group,state,optimal))
    return out


@torch.no_grad()
def branch_accuracy(net, probes):
    result = defaultdict(Counter)
    for i in range(0,len(probes),128):
        group = probes[i:i+128]
        acts = net(batch([r[1] for r in group]))[0].argmax(1).tolist()
        for (name, _, good),a in zip(group,acts):
            result[name].update(n=1, correct=int(a in good))
    return result


def score(net,validation,probes):
    rows = [r for i in range(0,len(validation.cases),32)
            for r in evaluate(net,Chunk(validation,validation.cases[i:i+32]),1)]
    return dict(wins=sum(r["outcome"]=="SUCCESS" for r in rows),branches=branch_accuracy(net,probes),rows=rows)


def main(seeds):
    OUT.mkdir(exist_ok=True)
    setup(71809)
    validation = Pool("validation_id",300,offset=1000)
    branch_states = probes(validation)
    for seed in seeds:
        name = f"double-v03-s{seed}"
        outfile = OUT / f"{name}.json"
        if outfile.exists():
            continue
        folder = SRC / "runs" / name
        config = read(folder/"run.json")["config"]
        setup(config["seed"])
        original = Network(config).cuda().eval()
        original.load_state_dict(torch.load(folder/"weights.pt",map_location="cuda",weights_only=True))
        control = None
        if seed==2:
            repeat, _ = train(config,ORIGINAL)
            diffs = {k:(v.float()-repeat.state_dict()[k].float()).abs().max().item()
                     for k,v in original.state_dict().items()}
            control = dict(tensors=len(diffs),max_abs_difference=max(diffs.values()))
            if control["max_abs_difference"] != 0:
                print("Архив и повтор расходятся",control,flush=True)
                original = repeat
        altered, training = train(config,without_coordinates)
        weights = OUT / f"{name}-without-coordinate.pt"
        torch.save(altered.state_dict(),weights)
        original_score = score(original,validation,branch_states)
        altered_score = score(altered,validation,branch_states)
        assert [r["map"] for r in original_score["rows"]] == [r["map"] for r in altered_score["rows"]]
        result = dict(model=name,config=config,repeated_control=control,original_sha256=digest(folder/"weights.pt"),
                      altered_sha256=digest(weights),validation_geometry_range=[1000,1299],final_episodes=1200,
                      altered_train_outcomes=training, original=original_score, without_coordinates=altered_score)
        outfile.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
        print(name,"validation",original_score["wins"],altered_score["wins"],"branches",original_score["branches"],
              altered_score["branches"],"control",control,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,choices=range(1,6),nargs="+",default=[2,1,3,4,5])
    main(parser.parse_args().seeds)
