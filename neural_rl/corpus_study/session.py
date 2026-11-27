"""Срок сессии и общий журнал экспериментов."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/E-0012"
DEADLINE = datetime.fromisoformat(json.loads((OUT/"session.json").read_text())["deadline_utc"]).timestamp()


def check_time():
    if time.time() >= DEADLINE:
        raise TimeoutError("Достигнут предел девяти часов")


@contextmanager
def locked(path):
    with path.open("a+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def log(event, **fields):
    record = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), event=event, **fields)
    with locked(OUT/".journal.lock"), (OUT/"experiments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
