import numpy as np
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import level,rows,save,digest
from .gates import HERE
from .symmetry import MOVES,ORDER,observation,raster


def main():
    src=HERE/"large.jsonl"
    checked=0
    for row in rows(src)[700:800:25]:
        rng=np.random.default_rng(int(row["geometry"][:10],16))
        environments=[]
        for j in range(8):
            record=dict(row,grid=raster(tuple(row["grid"]),j))
            environments.append(Environment(level(record)))
        for _ in range(35):
            e=environments[0]
            for j,other in enumerate(environments):
                assert other.view==observation(e.view,j)
                assert other.status==e.status
                checked+=1
            if e.status!=Status.RUNNING:break
            allowed=[a for a,m in enumerate(MOVES) if destination(e.view.window,(1,1),m) is not None]
            pick=int(rng.choice(allowed))
            for j,other in enumerate(environments):
                other.step(MOVES[ORDER[j][pick]])
    save(HERE/"rotation-check.json",dict(checks=checked,train_sha256=digest(src),indices=[700,725,750,775],
         transform_sha256=digest(HERE/"symmetry.py")))
    print("physical symmetry checks",checked,flush=True)


if __name__=="__main__":main()
