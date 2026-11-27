"""Воспроизводимый порядок подготовки выборок с отдельным генератором умеренного переноса."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sqlite3
import time

from maze.datasets import DATA, development_sets, geometry
from .corpus import DB, build, connect, selfcheck
from .ood_queue import build_parallel
from .session import OUT, check_time, locked, log
from .shift_guard import install
from .transfer import main as generate_shift


def run(workers):
    selfcheck()
    excluded = {geometry(c.level.grid) for cases in development_sets() for c in cases}
    excluded.update(x["geometry"] for x in json.loads((DATA / "test-manifest.json").read_text()).values())
    with connect() as db:
        install()
        with ProcessPoolExecutor(workers) as pool:
            build(db, pool, "validation_id", 60, excluded)
            for size in (300, 3000, 30000, 60000):
                build(db, pool, "train", size, excluded)
            build(db, pool, "validation_id", 3000, excluded)
        generate_shift()
        with ProcessPoolExecutor(workers) as pool:
            build(db, pool, "test_id", 6000, excluded)
    with locked(OUT / ".ood-writer.lock"):
        build_parallel(workers)


def resume_after_separate_shift(workers):
    """Продолжить закрытые выборки, если ранний генератор встретил защиту рецепта."""
    with sqlite3.connect(DB.as_uri()+"?mode=ro", uri=True) as db:
        while True:
            check_time()
            counts = dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))
            if counts.get("test_id", 0):
                return
            if (counts.get("train") == 60000 and counts.get("validation_id") == 3000
                    and counts.get("shift_development") == 1500):
                blocked = False
                with (OUT / "experiments.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        if '"geometry_retry"' in line and '"shift_development"' in line and line.endswith("\n"):
                            event = json.loads(line)
                            if event.get("split") == "shift_development":
                                blocked = True
                                break
                if blocked:
                    log("corpus_resumed_after_registered_shift", geometries=1500)
                    break
            time.sleep(20)
    run(workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume-after-separate-shift", action="store_true")
    args = parser.parse_args()
    if args.resume_after_separate_shift:
        resume_after_separate_shift(args.workers)
    else:
        run(args.workers)
