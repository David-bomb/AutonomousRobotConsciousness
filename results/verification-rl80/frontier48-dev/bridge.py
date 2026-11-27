"""Точка поставки load(bytes), привязанная к копиям зависимостей frontier48."""
from importlib import import_module

PACKAGE = "results.verification-rl80.frontier48-dev.frozen"


def frozen_import(name):
    if name == "results.E-0016.sequence.model":
        return import_module(PACKAGE + ".sequence.model")
    if name == "maze.interface":
        return import_module(name)
    raise ImportError(name)


# Меняется привязка импортов только в собственной копии адаптера.
# Исходный код выбора хода и реестр модулей остального проекта сохраняются.
adapter = import_module(PACKAGE + ".frontier_adapter")
adapter.import_module = frozen_import
load = adapter.load
