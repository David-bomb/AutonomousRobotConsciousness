from pathlib import Path
import ctypes as ct
import numpy as np

HERE=Path(__file__).resolve().parent
DLL=ct.CDLL(str(HERE/"landmarks.dll"))
DLL.landmark_features.argtypes=[np.ctypeslib.ndpointer(np.uint8,flags="C_CONTIGUOUS"),
      np.ctypeslib.ndpointer(np.int16,flags="C_CONTIGUOUS"),np.ctypeslib.ndpointer(np.float32),np.ctypeslib.ndpointer(np.float32)]
DLL.landmark_features.restype=None


def landmarks(board,meta):
    pair=np.empty((6,6,18),np.float32)
    action=np.empty((8,5,16),np.float32)
    DLL.landmark_features(board,meta,pair,action)
    return pair,action
