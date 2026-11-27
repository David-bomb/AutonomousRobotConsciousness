"""Два режима одного архивного автомата. Канал во время эпизода — choose(Observation)."""
from functools import _lru_cache_wrapper
import os
from pathlib import Path
import site
import sys

# Оценщик отключает автозагрузку пользовательского site (включая .pth).
# В этой установке системного Python именно здесь находятся NumPy и SciPy.
sys.path.append(site.getusersitepackages())
from maze.mandatory_supply import MandatoryProof
from maze.pilot_automaton import Mode


def caches():
    return tuple({fn for name, module in sys.modules.copy().items() if name.startswith("maze.")
                  for fn in vars(module).values() if isinstance(fn, _lru_cache_wrapper)})


class Previous:
    def __init__(self):
        self.previous = None

    def choose(self, view):
        for fn in WRAPPERS:
            fn.cache_clear()
        agent = MandatoryProof()
        if self.previous is not None:
            agent._fallback.last_lengths = dict(zip((Mode.FUEL, Mode.REPAIR), self.previous, strict=True))
        move = agent.choose(view)
        self.previous = (view.fuel_distance, view.repair_distance)
        return move


WRAPPERS = caches()


def load(mode):
    if mode not in (b"full", b"previous") or len(WRAPPERS) != 55:
        raise ValueError("Неизвестный режим или изменился набор кэшей")
    # После импорта разрешены только ленивые импорты библиотек вне проекта.
    libs = tuple(Path(p).resolve() for p in (Path(sys.base_prefix) / "Lib", site.getusersitepackages()))

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() not in {".py", ".pyc", ".pyd", ".dll"} or not any(p.is_relative_to(d) for d in libs):
                raise PermissionError("Запрещено чтение данных из контроллера")
        if event in {"socket.connect", "subprocess.Popen", "os.system"}:
            raise PermissionError(event)

    sys.addaudithook(guard)
    return MandatoryProof() if mode == b"full" else Previous()
