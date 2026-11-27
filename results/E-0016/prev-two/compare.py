"""Одновременные независимые проверки, по одному потоку на экземпляр модели."""
from concurrent.futures import ProcessPoolExecutor
import sys
from .physical import evaluate
from .common import HERE,read


def run(job):
    return evaluate(*job)


def rank(tag):
    result=read(HERE/"dev"/(tag+"-honest.json"))["summary"]
    return result["wins"],min(result["tiers"])


if __name__=="__main__":
    tags=sys.argv[2:]
    modes=["honest","ablated"] if sys.argv[1]=="both" else ["honest"]
    with ProcessPoolExecutor(2) as workers:
        list(workers.map(run,[(tag,mode) for tag in tags for mode in modes]))
