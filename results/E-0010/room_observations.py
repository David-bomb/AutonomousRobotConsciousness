"""Сколько предполагаемых планировок поддерживало действие в первом расхождении."""
import argparse
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import PhysicalRoutes, read, transition
from maze.environment import Environment
from maze.mission_order import MissionOrderReserve, RoomMission
from maze.room_roads import RoomPassages, room_predicts_memory, room_votes
from maze.street_trunk import StreetTrunk


class CaptureRoom:
    capture = False
    evidence = None

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        chosen=super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        if self.capture:
            scores=room_votes(view,readings,old,prior,sites,True)
            self.evidence=dict(service="FR"[kind],default=default.name,
                               options=[m.name for m in options],
                               weights={m.name:float(w) for m,w in scores[kind].items()} if scores else {})
        return chosen


class ObservedRoom(CaptureRoom, RoomPassages):
    pass


class ObservedRoomMission(CaptureRoom, RoomMission):
    pass


def measure(level,base,child,i,mission=False):
    clear_episode_caches()
    env=Environment(level)
    bots=(MissionOrderReserve(),ObservedRoomMission()) if mission else (StreetTrunk(),ObservedRoom())
    for j in range(i+1):
        bots[1].capture = j==i
        moves=tuple(bot.choose(env.view) for bot in bots)
        assert all(move==MOVES[int(row["actions"][j])] for move,row in zip(moves,(base,child)))
        if j<i:
            assert moves[0]==moves[1]
            env.step(moves[0])
    routes=PhysicalRoutes(level)
    state=env.position,env.fuel,env.health
    result = dict(step=i+1,pos=env.position,stocks=[env.fuel,env.health],
                lengths=[env.view.fuel_distance,env.view.repair_distance],
                moves=[m.name for m in moves],safe=[transition(routes,state,m,env.limit-i-1) for m in moves],
                evidence=bots[1].evidence)
    if mission:
        result["room_predicts_memory"] = bool(room_predicts_memory(env.view.memory))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", action="store_true")
    args = parser.parse_args()
    path=HERE/("room-mission-observations.jsonl" if args.mission else "room-observations.jsonl")
    saved={r["map"]:r for line in path.read_bytes().splitlines() if (r:=json.loads(line))} if path.exists() else {}
    base=read("mission-order-reserve" if args.mission else "street-trunk")
    child_path=HERE/("room-mission.jsonl" if args.mission else "room-passages.jsonl")
    children={r["map"]:r for line in child_path.read_bytes().splitlines() if (r:=json.loads(line))}
    with path.open("a",encoding="utf8") as out:
        for split,name,_,_,level in cases():
            if name not in children:
                continue
            old,new=base[name],children[name]
            diff=next((i for i,(a,b) in enumerate(zip(old["actions"],new["actions"])) if a!=b),None)
            if diff is None:
                assert old["actions"]==new["actions"] and old["status"]==new["status"]
                continue
            statuses=[old["status"],new["status"]]
            if name in saved:
                assert saved[name]["step"]==diff+1 and saved[name]["statuses"]==statuses
                continue
            row=dict(map=name,split=split,statuses=statuses,**measure(level,old,new,diff,args.mission))
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print("Проверено записанных поездок",len(children),flush=True)


if __name__=="__main__":
    main()
