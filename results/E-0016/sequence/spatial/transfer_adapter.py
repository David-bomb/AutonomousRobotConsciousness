from importlib import import_module

def load(checkpoint):
    return import_module("results.E-0016.sequence.spatial.transfer_model").load(checkpoint)
