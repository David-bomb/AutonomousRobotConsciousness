"""Construct two valid mazes with identical full initial Observation and disjoint winning actions."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "University/Diplom"))
from maze.environment import Environment, Status
from maze.interface import Move
from maze.levels import Level
from scripts.audit_automaton import Reachability


def board():
    cells = [list("#"*13) for _ in range(12)]
    for x in range(2,11):
        cells[9][x]="."
    for y in range(5,10):
        cells[y][2]="."
    for x in range(2,7):
        cells[5][x]="."
    for y,kind in [(9,"A"),(5,"F"),(4,"R"),(3,"F"),(2,"R"),(1,"B")]:
        cells[y][6]=kind
    return ["".join(r) for r in cells]


def inspect(rows, side):
    env=Environment(Level.parse("\n".join([side,"12 13",*rows])))
    initial=env.view
    possible=Reachability(env.level)
    directions=[m.name for m,p in possible.edges[env.position]
                if possible.possible(p,*possible.after(p,env.fuel,env.health),env.limit-1)]
    first=Move.WEST if side=="left" else Move.EAST
    inward=Move.EAST if side=="left" else Move.WEST
    route=[first]*4+[Move.NORTH]*4+[inward]*4+[Move.NORTH]*4
    for m in route:
        env.step(m)
    assert env.status==Status.SUCCESS
    return initial,dict(grid=rows,viable_first=directions,witness=[m.name for m in route],
                        physical_status=env.status.name,steps=env.steps)


if __name__=="__main__":
    left=board()
    a, one=inspect(left,"left")
    b, two=inspect([r[::-1] for r in left],"right")
    assert a==b and set(one["viable_first"]).isdisjoint(two["viable_first"])
    result=dict(scope="constructive example in project physics; deliberately constructed, outside benchmark",
                observation=asdict(a),left=one,right=two)
    (HERE/"paired-observations.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(a.position,a.fuel_distance,a.repair_distance,one["viable_first"],two["viable_first"])
