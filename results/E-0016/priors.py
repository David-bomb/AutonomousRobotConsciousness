"""Таблица вероятности дороги, оценённая только по полным геометриям train."""
import numpy as np
from .rich_policy import contexts


def fit_prior(train):
    total=np.full(648,2.,np.float64)
    opened=np.ones(648,np.float64)
    for r in train:
        board=np.asarray([list(row.encode("ascii")) for row in r["grid"]],np.uint8)
        roads=(board!=ord("#"))
        h,w=board.shape
        side=np.pad(roads.astype(np.int32)+1,1,constant_values=1)
        neighbors=(side[:-2,1:-1],side[2:,1:-1],side[1:-1,:-2],side[1:-1,2:])
        parity=contexts(h,w)*81
        for mask in range(16):
            codes=parity.copy()
            for bit,cells in enumerate(neighbors):
                if mask & (1<<bit):
                    codes+=3**bit*cells
            total+=np.bincount(codes.ravel(),minlength=648)
            opened+=np.bincount(codes[roads],minlength=648)
    return (opened/total).astype(np.float32)
