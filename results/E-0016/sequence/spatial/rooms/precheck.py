"""Полные поездки с зарегистрированными весами выхода до терминального обучения."""
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import torch

from ....data import digest, read, rows, save
from ....dev import evaluate
from .cells import HERE
from .model import load
from .physical import ROOT, summarize


def init(data):
    global controller
    controller = load(data)


def journey(row):
    return evaluate(controller, [row])[1][0]


def main():
    folder = ROOT / "runs/rooms3000"
    lock = read(folder / "locked-before-dev.json")
    trained = folder / "rl.pt"
    assert digest(trained) == lock["weights_sha256"]
    before = lock["pre_RL"]["weights"]
    assert before == read(folder / "config.json")["choices"][1]
    weights = folder / "pre.pt"
    if not weights.exists():
        pack = torch.load(trained, weights_only=True, map_location="cpu")
        pack["linear"] = torch.tensor(before, dtype=torch.float64)
        torch.save(pack, weights)
    data = weights.read_bytes()
    assert torch.load(BytesIO(data), weights_only=True, map_location="cpu")["linear"].tolist() == before
    bank = HERE / "independent.jsonl"
    assert digest(bank) == read(HERE / "independent-bank.json")["sha256"]
    cache = ROOT / "cache/rooms-independent-pre"
    cache.mkdir(exist_ok=True)
    cfg = dict(weights=digest(weights), source=digest(HERE / "precheck.py"), bank=digest(bank))
    if (cache / "config.json").exists():
        assert read(cache / "config.json") == cfg
    else:
        save(cache / "config.json", cfg)
    maps = rows(bank)
    with ProcessPoolExecutor(2, initializer=init, initargs=(data,)) as pool:
        for ep in pool.map(journey, [r for r in maps if not (cache / (r["name"] + ".json")).exists()]):
            save(cache / (ep["map"] + ".json"), ep)
    episodes = [read(cache / (r["name"] + ".json")) for r in maps]
    result = summarize(episodes)
    save(folder / "independent-pre.json", dict(config=cfg, summary=result, episodes=episodes))
    print(result, flush=True)


if __name__ == "__main__":
    main()
