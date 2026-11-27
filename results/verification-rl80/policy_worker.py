"""Процесс одного эпизода. На стандартном входе веса, затем только Observation."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maze.interface import Move, Observation


def payload(source):
    size = int(source.readline())
    data = source.read(size)
    if len(data) != size:
        raise ValueError("Неполное содержимое")
    return data


def main():
    source, sink = sys.stdin.buffer, sys.stdout
    adapter, checkpoint = payload(source), payload(source)
    module = ModuleType("rl80_frozen_adapter")
    sys.modules[module.__name__] = module
    with redirect_stdout(sys.stderr):
        exec(compile(adapter, "<rl80_frozen_adapter>", "exec"), module.__dict__)
        controller = module.load(checkpoint)
    for line in source:
        fields = json.loads(line)
        for key in ("memory", "window", "position"):
            fields[key] = tuple(fields[key])
        if fields["last_move"] is not None:
            fields["last_move"] = Move[fields["last_move"]]
        with redirect_stdout(sys.stderr):
            action = controller.choose(Observation(**fields))
        if type(action) is not Move:
            raise TypeError("choose должен вернуть один maze.interface.Move")
        print(action.name, file=sink, flush=True)


if __name__ == "__main__":
    main()
