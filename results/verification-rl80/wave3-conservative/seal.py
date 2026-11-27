"""Проверить сохранённую волну и выпустить общую квитанцию для директора."""
import json
import subprocess
import sys

from run import HERE, ID, OUT, SPLITS, WEIGHTS, file_hash, object_hash, read, write_new


LOCK = "dc2f9a56764395522d7181070c2a495e1afb69f6643f4191956b9722c03f5bd5"


def main():
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "run.py"), "verify", LOCK], check=True)
    registration = read(HERE / "registered.json")
    details = read(HERE / "receipt.json")
    assert registration["candidate"] == details["candidate"] == ID
    assert details["checkpoint_sha256"] == WEIGHTS and details["lock_sha256"] == LOCK
    assert details["episodes"] == details["empty_registers_checked"] == 300
    assert details["unique_geometries"] == details["unique_ids"] == 300 and not details["violations"]
    rows = [json.loads(line) for line in (OUT / "episodes.jsonl").read_bytes().splitlines()]
    assert all(row["policy_calls"] == row["steps"] for row in rows)
    assert all(details["by_split"][group]["maps"] == 100 for group in SPLITS)
    assert list(details["successes"]) == list(SPLITS)

    # PID назначает Windows. Остальные поля определены сценарием, весами и средой.
    deterministic = [{key: value for key, value in row.items() if key != "worker_pid"} for row in rows]
    public = dict(candidate=ID, weights_sha256=WEIGHTS, threshold=0.9, lock_sha256=LOCK,
                  adapter_sha256=file_hash(HERE / "frozen/results/E-0016/prev-two/cautious_policy.py"),
                  receipt_file_sha256=file_hash(HERE / "receipt.json"),
                  deterministic_episodes_sha256=object_hash(deterministic),
                  successes=details["successes"], passed=details["passed"],
                  checked_episodes=300, policy_calls_equal_physical_steps=True, violations=[])
    target = HERE / "aggregate.json"
    if target.exists():
        assert read(target) == public
    else:
        write_new(target, public)
    print(json.dumps(public, ensure_ascii=False))


if __name__ == "__main__":
    main()
