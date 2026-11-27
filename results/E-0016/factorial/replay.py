"""Заново обучает техническую R1C1 в новом каталоге, сверяет веса и все поездки."""
from . import run
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    out = run.OUT
    reg = run.read(run.REG)
    index = json.loads(gzip.decompress(run.INDEX.read_bytes()))
    run.torch.set_num_threads(2)
    run.torch.set_num_interop_threads(1)
    run.learn.DB = run.SNAPSHOT
    run.perception.SIDE = reg["input_canvas"]
    run.training.torch = run.CpuTorch()
    run.training.batch = run.rollout.batch = run.batch_cpu
    run.training.Stage = run.RecordedStage
    cfg = reg["configs"][0]
    arm = "R1C1"
    location = Path("technical")/str(cfg["seed"])/arm
    prev = run.read(out/location/"training.json")
    with TemporaryDirectory(dir=out) as tmp:
        run.OUT = Path(tmp)
        run.run_one("technical",arm,cfg,reg,index)
        repeat = run.read(run.OUT/location/"training.json")
        assert prev["initial"] == repeat["initial"] and prev["final"] == repeat["final"]
        assert prev["history"] == repeat["history"]
        old = run.digest(out/location/"validation.json")
        new = run.digest(run.OUT/location/"validation.json")
        assert old == new
    run.save(out/"replay.json", dict(seed=cfg["seed"], arm=arm, phase="technical",
                                    initial_sha256=prev["initial"], final_tensor_sha256=repeat["final"],
                                    original_file_sha256=prev["weights_sha256"],
                                    repeat_file_sha256=repeat["weights_sha256"], train_history_identical=True,
                                    validation_sha256=old, identical_validation_episodes=240,
                                    replay_script_sha256=run.digest(out/"replay.py")))
    print("Identical final tensors, training history and all 240 episodes")


if __name__ == "__main__":
    main()
