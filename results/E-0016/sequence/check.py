"""Проверка размерностей, граничного обслуживания и разметки будущих датчиков."""
from copy import copy
import numpy as np
from maze.environment import Environment,Status
from maze.grid import destination
from maze.levels import Level
from .bank import HERE
from .model import LIB
from ..data import level,rows,save
from ..policy import MOVES
from ..teacher import Teacher


def main():
    tiny=Level.parse("проверка\n2 6\n#############\n#A.F.R.F.R.B#\n#############")
    env=Environment(tiny)
    grid="".join(tiny.grid).encode()
    w=len(tiny.grid[0])
    targets=np.array([w+x for x in (11,3,7,5,9)],np.int32)
    out=np.zeros((8,92),np.float32)
    LIB.sequences(grid,grid,w,len(tiny.grid),w+1,targets,2,6,2,4,out)
    orders=out[1,:20].reshape(5,4)
    assert orders[0].tolist()==[10,-8,300,0]
    assert orders[3,2]==10 and orders[3,1]==0 and orders[4,3]==0
    for _ in range(10):
        env.step(MOVES[1])
    assert env.status==Status.SUCCESS and (env.steps,env.refuels,env.repairs)==(10,2,2)
    checked=0
    for row in rows(HERE/"train.jsonl")[:20]:
        env=Environment(level(row))
        oracle=Teacher(env.level)
        for t in range(12):
            v=env.view
            for m in MOVES:
                q=destination(v.memory,v.position,m)
                if q is None:
                    continue
                alt=copy(env)
                alt._memory=copy(env._memory)
                following=alt.step(m)
                assert min(d[q] for d in oracle.dist[:2])-v.fuel_distance==following.fuel_distance-v.fuel_distance
                assert min(d[q] for d in oracle.dist[2:4])-v.repair_distance==following.repair_distance-v.repair_distance
                checked+=1
            best,_=oracle.labels(env)
            env.step(MOVES[int(np.flatnonzero(best)[0])])
            if env.status!=Status.RUNNING:
                break
        oracle.remaining.cache_clear()
    save(HERE/"checks.json",dict(known_order_rows=orders.tolist(),physically_checked_gradient_actions=checked,
                                 toy=dict(outcome="SUCCESS",steps=10,refuels=2,repairs=2)))
    print(checked,"действий, оба датчика совпали",flush=True)


if __name__=="__main__":
    main()
