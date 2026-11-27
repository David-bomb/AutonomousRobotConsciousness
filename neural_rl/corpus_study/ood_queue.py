"""Параллельный поиск допустимых рисунков для больших отложенных карт."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from maze.datasets import DATA, development_sets, geometry
from .corpus import TARGETS, connect, make
from .report import write_text
from .session import OUT, check_time, locked, log


def candidate(task):
    index, first = task
    attempt = first
    causes = Counter()
    while True:
        check_time()
        # Первые 1000 совпадают с исходной регистрацией. Высокий разряд
        # расширяет пространство зёрен без пересечения с соседней картой.
        band, offset = divmod(attempt, 1000)
        result = make(("test_ood", index, 10**13 * band + offset))
        if not isinstance(result, str):
            return result, attempt, causes
        causes[result] += 1
        attempt += 1


def build_parallel(workers):
    size = TARGETS["test_ood"]
    excluded = {geometry(c.level.grid) for cases in development_sets() for c in cases}
    excluded.update(x["geometry"] for x in json.loads((DATA / "test-manifest.json").read_text()).values())
    plan = OUT / "ood-generation.json"
    recipe = dict(target=size, split="test_ood",
                  generator_sha256=sha256(Path(__file__).with_name("corpus.py").read_bytes()).hexdigest(),
                  scheduler_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), initial_attempts=1000, seed_band=10**13,
                  order="increasing geometry index and candidate seed, first certified unique geometry")
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != recipe:
        raise ValueError("Изменилось зарегистрированное правило продолжения OOD")
    write_text(plan, json.dumps(recipe, ensure_ascii=False, indent=2)+"\n")
    with connect() as db, ProcessPoolExecutor(workers) as pool:
        existing = {r[0] for r in db.execute("SELECT idx FROM geoms WHERE split='test_ood'")}
        missing = [i for i in range(size) if i not in existing]
        log("ood_queue_started", workers=workers, existing=len(existing), **recipe)
        rejects = Counter()
        extended = 0
        for completed, (i, found) in enumerate(zip(missing, pool.map(candidate, ((i, 0) for i in missing))), 1):
            result, attempt, causes = found
            rejects.update(causes)
            while True:
                if result[2] not in excluded:
                    try:
                        with db:
                            db.execute("INSERT INTO geoms VALUES (?,?,?,?,?,?,?,?)", result)
                        break
                    except sqlite3.IntegrityError:
                        if db.execute("SELECT 1 FROM geoms WHERE split='test_ood' AND idx=?", (i,)).fetchone():
                            raise RuntimeError("Обнаружен второй писатель той же отложенной карты")
                rejects["Повтор рисунка стен"] += 1
                result, attempt, causes = candidate((i, attempt+1))
                rejects.update(causes)
            if attempt >= 1000:
                extended += 1
                log("ood_extended_seed", index=i, attempts=attempt+1, accepted_seed=result[5])
            count = len(existing) + completed
            if count % 100 == 0 or count == size:
                log("ood_queue_progress", geometries=count, certified=4*count, extended=extended,
                    rejection_reasons=dict(rejects))
        total, = db.execute("SELECT count(*) FROM geoms WHERE split='test_ood'").fetchone()
        if total != size:
            raise ValueError("Неполная отложенная выборка")
        log("corpus_milestone", split="test_ood", geometries=total, certified=4*total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    with locked(OUT / ".ood-writer.lock"):
        build_parallel(parser.parse_args().workers)
