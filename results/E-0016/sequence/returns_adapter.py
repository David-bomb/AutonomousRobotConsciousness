"""Передача замороженных весов модели, обученной на вариантах продолжений."""
from importlib import import_module

def load(checkpoint):
    return import_module("results.E-0016.sequence.value_model").load(checkpoint)
