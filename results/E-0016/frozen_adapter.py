"""Точка входа отдельного процесса оценщика; получает только байты весов."""
from importlib import import_module


def load(checkpoint: bytes):
    return import_module("results.E-0016.adapter").from_bytes(checkpoint)
