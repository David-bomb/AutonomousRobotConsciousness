"""Передача байтов закреплённых весов неизменённому адаптеру recovery800."""
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

EXPECTED = "020574084dcfaa3efffca38d0c057419e0e90307548f06bfb62acad5be47946f"


def load(checkpoint: bytes):
    module = import_module("results.verification-rl80.recovery800-dev.frozen.adapter")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        path.write_bytes(checkpoint)
        return module.Controller(path, EXPECTED)
