from importlib import import_module


def load(data):
    return import_module("results.E-0016.sequence.spatial.reconstruct.dagger_model").load(data)
