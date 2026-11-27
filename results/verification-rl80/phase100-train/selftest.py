"""Граничные случаи передачи, расхода последней единицы и лимита."""
from driver import Environment, Exact, Level, Move, Status, boundary, play, readings
from scripts.audit_automaton import Reachability
from verify import check_witness


def line(cells,stocks):
    wall="#"*len(cells)
    return Level.parse("\n".join(["Синтетика",stocks,wall,cells,wall]))


def test():
    early=line("#ABFRFR#","1 1")
    env=Environment(early)
    route=Exact(early).route(env)
    assert route==b"\x01"
    prev=play(env,route)
    assert env.fuel==env.health==0 and boundary(env,prev)["kind"]=="early_goal"

    both=line("#AFR.BFR#","1 2")
    env=Environment(both)
    shortest=Exact(both).route(env)
    assert shortest==b"\x01"*4
    first_pair=readings(env.view)
    env.step(Move.EAST)
    assert env.fuel==100 and env.health==1
    previous=readings(env.view)
    env.step(Move.EAST)
    snap=boundary(env,previous)
    assert snap["kind"]=="handoff" and snap["steps"]==2 and snap["previous"]==previous
    assert previous!=first_pair and previous!=readings(env.view)
    assert env.fuel==99 and env.health==100
    route=Exact(both).route(env)
    check_witness(env,"".join(map(str,route)),Reachability(both))

    env=Environment(both,limit=2)
    prev=play(env,shortest[:2])
    assert boundary(env,prev)["kind"]=="terminal_on_second_service" and env.status==Status.LIMIT
    assert Exact(both).route(env) is None
    env=Environment(both,limit=3)
    prev=play(env,shortest[:2])
    assert boundary(env,prev)["kind"]=="handoff" and Exact(both).route(env) is None
    check_witness(env,None,Reachability(both))
    print("OK: ранний B, пополнение на последней единице, предыдущие датчики и две границы лимита.")


if __name__=="__main__":
    test()
