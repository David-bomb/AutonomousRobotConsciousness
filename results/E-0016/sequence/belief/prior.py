"""Вероятности неизвестных клеток для четырёх семейств учебных геометрий."""
from collections import Counter
from pathlib import Path
import numpy as np
from ...data import rows,save,digest
from ...priors import fit_prior

HERE=Path(__file__).resolve().parent
FAMILIES=("corridors","branching","rooms","loops")


def main():
    p=HERE.parent/"train.jsonl"
    bank=rows(p)
    stats=np.stack([fit_prior([r for r in bank if r["family"]==f]) for f in FAMILIES])
    out=HERE/"prior.npy"
    np.save(out,stats)
    save(HERE/"prior.json",dict(train_sha256=digest(p),groups=dict(Counter(r["family"] for r in bank)),
         families=FAMILIES,shape=list(stats.shape),sha256=digest(out)))


if __name__=="__main__":
    main()
