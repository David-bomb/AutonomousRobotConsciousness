"""Выполнить исходный frozen_adapter.py с зарегистрированными копиями его модулей."""
from importlib import import_module
import sys


def load(checkpoint: bytes):
    snapshot = import_module("results.verification-rl80.wave1.frozen")
    sys.modules["results.E-0016"] = snapshot
    entry = import_module("results.E-0016.frozen_adapter")
    return entry.load(checkpoint)
