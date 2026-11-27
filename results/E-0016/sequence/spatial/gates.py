import ctypes as ct
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NATIVE = ct.CDLL(str(HERE/"gates.dll"))
NATIVE.gate_features.argtypes = [np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS"), ct.c_int, ct.c_int, ct.c_int,
    np.ctypeslib.ndpointer(np.float32), np.ctypeslib.ndpointer(np.int16)]
NATIVE.gate_features.restype = None


def describe(board, meta, stocks):
    slack = np.asarray(stocks) - meta[6:8]
    urgent = int(slack.argmin())
    x = np.empty((8, 8, 32), np.float32)
    ref = np.empty((8, 8, 4), np.int16)
    NATIVE.gate_features(board, meta, urgent, int(slack[urgent]), int(slack[1-urgent]), x, ref)
    return x, ref, urgent
