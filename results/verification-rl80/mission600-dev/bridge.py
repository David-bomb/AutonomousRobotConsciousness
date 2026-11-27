"""Загрузить штатный адаптер mission600 и все используемые им локальные модули из снимка."""
from importlib import import_module
import sys


def load(checkpoint: bytes):
    sys.modules["results.E-0016"] = import_module("results.verification-rl80.mission600-dev.frozen")
    return import_module("results.E-0016.sequence.adapter").load(checkpoint)
