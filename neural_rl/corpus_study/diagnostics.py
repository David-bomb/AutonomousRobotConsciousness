"""Двадцать парных и ресурсных проверок, изготовленных в штатном формате уровней."""
from collections import Counter
import gzip
import hashlib
import json
import random
import sqlite3

from maze.datasets import DATA, development_sets, geometry
from maze.environment import Environment,Status
from maze.interface import Move
from maze.levels import Level
from .corpus import DB,MOVES,certify,edges,path,tree
from .session import OUT,check_time,log
from .report import write_text

NAMES=("Прямая дорога","U-образный обход","Возврат из тупика","Одинаковое окно после разных поездок",
       "Разные карты памяти","Выбор при разных запасах","Близость по координатам и по дороге",
       "Смена ближайшей станции","Равные длины до двух станций","Запас в точности до сервиса",
       "Неразличимая развилка","Рядом финиш при недоступном по запасу сервисе","Лишний заезд",
       "Цепочка заправок","Обязательное повторное посещение","Циклы между сервисами",
       "Прибытие на последней единице","Отсутствующий или недостижимый сервис",
       "Повороты и прямоугольные уровни","Лимит среды и разрез сбора переходов")
PAIRED=(4,5,6,11,19)
FOLDER=OUT/"diagnostics"


class Drawing:
    def __init__(self):
        self.roads={(0,0)}
        self.start=(0,0)
        self.goal=(8,0)
        self.sites={}
        self.stocks=[100,100]
        self.via=[]
        self.probe=[]

    def line(self,*corners):
        for (x,y),(u,v) in zip(corners,corners[1:]):
            if x!=u and y!=v: raise ValueError("Наклонная линия в шаблоне")
            for i in range(max(abs(u-x),abs(v-y))+1):
                self.roads.add((x+i*((u>x)-(u<x)),y+i*((v>y)-(v<y))))


def drawing(group,variant,rng):
    d=Drawing()
    a=rng.randint(4,7)
    b=rng.randint(3,6)
    length=rng.randint(12,20)
    d.goal=(length,0)
    if group in (4,5,11):
        d.goal=(0,-b-2)
        d.line((-a,0),(a,0),(a,-b),(-a,-b),(-a,0))
        d.line((0,-b),d.goal)
        good=-1 if group==5 or variant==0 else 1
        d.roads.remove((-good*(a//2),-b))
        d.sites={(0,-b):"F",(0,-b-1):"R"}
        if group==11:
            d.stocks[0]=2*a+b
        else:
            sides=(-1,1) if group==4 else ((-1,) if variant==0 else (1,))
            for side in sides:
                d.probe.extend([(side*a,0),(side*a,-b),(side*(a//2+1),-b),(side*a,-b),(side*a,0),d.start])
            d.probe.extend([(1,0),d.start])
    elif group==2:
        d.goal=(3,0)
        d.line(d.start,(-a,0),(-a,b),(3,b),d.goal)
        d.sites={(-a,b):"R"}
    elif group in (3,6,13,15):
        d.line((-a-2,0),d.goal)
        d.line((4,0),(4,b))
        symbol="R" if group==6 and rng.randrange(2) else "F"
        idx=int(symbol=="R")
        d.sites[(-a,0)]=symbol
        if group in (3,15) or (group==6 and variant==1):
            d.stocks[idx]=a+1
            d.via=[(-a,0)]
        if group==15:
            d.sites[(-a-2,0)]="R"
            d.stocks[1]=a+2
            d.via=[(-a-2,0)]
    elif group==7:
        d.goal=(-a-2,0)
        d.line(d.goal,(a+2,0),(a+2,-2),(0,-2))
        d.sites={(-a,0):"F",(0,-2):"F"}
    elif group in (8,9):
        right=a+int(group==8)
        d.goal=(right+3,0)
        d.line((-a,0),d.goal)
        d.sites={(-a,0):"F",(right,0):"F"}
    elif group in (10,12,17):
        d.line((-a-2,0),d.goal)
        if group==10:
            d.sites[(1,0)]="F"
            d.stocks[0]=1
        elif group==12:
            d.goal=(rng.randint(2,4),0)
            d.sites[(-a-2,0)]="F"
            d.stocks[0]=d.goal[0]
        elif variant:
            d.sites[(b,0)]="R"
            d.stocks[1]=b
        else:
            d.stocks=[length,length]
    elif group==14:
        width=rng.randint(22,30)
        rows=rng.randint(10,12)
        route=[d.start]
        for y in range(rows):
            end=width if y%2==0 else 0
            while route[-1][0]!=end:
                x,h=route[-1]
                route.append((x+(1 if end else -1),h))
            if y<rows-1:
                route.extend([(end,2*y+1),(end,2*y+2)])
        d.roads=set(route)
        d.goal=route[-1]
        for i in range(40,len(route)-1,75): d.sites[route[i]]="F"
        for i in range(65,len(route)-1,75): d.sites[route[i]]="R"
        d.stocks=[44,69]
    elif group in (16,20):
        d.line(d.start,d.goal)
        d.sites={(1,0):"F",(2,0):"R"}
        d.stocks=[100,100]
        if group==16: d.probe=[(1,0)]
    elif group==18:
        d.line(d.start,d.goal)
        d.sites[(a,0)]="R"
        if variant:
            isolated=(length+6,-3)
            d.roads.add(isolated)
            d.sites[isolated]="F"
    elif group==19:
        d.goal=(length,b)
        d.line(d.start,(a,0),(a,b),d.goal)
        d.line((a,b),(0,b))
        d.sites={(a,0):"F",(a,b):"R"}
    else:
        d.line(d.start,d.goal)
    return d


def prepare(group,index,bank,attempt):
    paired=group in PAIRED
    number=index//2 if paired else index
    seed=(10**8 if bank=="train" else 2*10**8)+group*100000+number*100+attempt
    rng=random.Random(seed)
    d=drawing(group,index%2,rng)
    # Ответвление за основной задачей даёт самостоятельные рисунки стен, сохраняя проверяемое свойство.
    top=min(y for x,y in d.roads if (x,y) not in d.sites or group!=18)
    x,y=min((p for p in d.roads if p[1]==top),key=lambda p:p[0])
    for _ in range(rng.randint(4,9)):
        stride=rng.randint(1,3)
        d.line((x,y),(x,y-2),(x+stride,y-2))
        x,y=x+stride,y-2
    rotation=(rng.randrange(4)+(index%2 if group==19 else 0))%4
    reflect=rng.randrange(2)
    def orient(p):
        x,y=p
        if reflect: x=-x
        for _ in range(rotation): x,y=-y,x
        return x,y
    rotated=[orient(p) for p in d.roads]
    left=min(p[0] for p in rotated)-rng.randint(1,3)
    top=min(p[1] for p in rotated)-rng.randint(1,3)
    width=max(p[0] for p in rotated)-left+rng.randint(2,4)
    height=max(p[1] for p in rotated)-top+rng.randint(2,4)
    def pos(p):
        x,y=orient(p)
        return x-left,y-top
    cells=[["#"]*width for _ in range(height)]
    for p in d.roads:
        x,y=pos(p); cells[y][x]="."
    for p,letter in [*d.sites.items(),(d.start,"A"),(d.goal,"B")]:
        x,y=pos(p); cells[y][x]=letter
    name=f"diag-{bank}-{group:02}-{index:02}"
    source="\n".join([name," ".join(map(str,d.stocks)),*map("".join,cells)])
    level=Level.parse(source)
    grid=tuple("".join("#" if c=="#" else "." for c in row) for row in level.grid)
    graph=edges(grid)
    def moves_for(points):
        current=level.start
        moves=[]
        for p in points:
            next_pos=pos(p)
            route=path(tree(graph,current)[0],next_pos)
            moves.extend(MOVES.index(Move((v[0]-u[0],v[1]-u[1]))) for u,v in zip(route,route[1:]))
            current=next_pos
        return "".join(map(str,moves))
    moves=moves_for([*d.via,d.goal])
    proof=certify(level,moves)
    if not proof: raise ValueError("Не удалось подтвердить диагностику "+name)
    prefix=moves_for(d.probe)
    env=Environment(level)
    for code in prefix: env.step(MOVES[int(code)])
    if env.status!=Status.RUNNING: raise ValueError("Проверочная позиция после завершения")
    # Проверки непосредственно заявленных граничных условий.
    if group==10: assert env.fuel==env.view.fuel_distance==1
    if group==11: assert env.fuel==env.view.fuel_distance
    if group==12: assert env.fuel<env.view.fuel_distance
    if group in (7,8,9):
        sites=[pos(p) for p,s in d.sites.items() if s=="F"]
        lengths=tree(graph,level.start)[1]
        if group==7:
            near=min(sites,key=lambda p:max(abs(p[0]-level.start[0]),abs(p[1]-level.start[1])))
            assert lengths[near]>env.view.fuel_distance
        elif group==9:
            assert lengths[sites[0]]==lengths[sites[1]]==env.view.fuel_distance
        else:
            following=pos((1,0))
            dx,dy=following[0]-level.start[0],following[1]-level.start[1]
            env.step(Move((dx,dy)))
            second=tree(graph,env.position)[1]
            assert min(sites,key=lengths.get)!=min(sites,key=second.get)
    if group==14: assert proof["refuels"]>=2 and proof["repairs"]>=2
    if group==17 and index%2==0: assert proof["final_fuel"]==proof["final_health"]==0
    if group==18: assert env.view.fuel_distance is None
    limit_steps=None
    if group==20:
        walk=moves_for([(1,0),(2,0)])
        for code in walk: env.step(MOVES[int(code)])
        assert env.status==Status.RUNNING
        forward=MOVES[int(walk[-1])]
        dx,dy=forward.value
        backward=Move((-dx,-dy))
        while env.status==Status.RUNNING:
            env.step(backward if env.position==pos((2,0)) else forward)
        assert env.status==Status.LIMIT
        limit_steps=env.steps
    return dict(name=name,group=group,bank=bank,pair=number,seed=seed,source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),
                geometry=geometry(level.grid),proof=proof,moves=moves,prefix=prefix,limit_steps=limit_steps)


def probe_view(row):
    env=Environment(Level.parse(row["source"]))
    for code in row["prefix"]: env.step(MOVES[int(code)])
    return env.view


def rotate_scenario(row,index):
    _,stocks,*grid=row["source"].splitlines()
    name=f"diag-{row['bank']}-19-{index:02}"
    rotated=["".join(line) for line in zip(*grid[::-1])]
    source="\n".join([name,stocks,*rotated])
    level=Level.parse(source)
    def rotate_moves(codes):
        values=(MOVES[int(ch)].value for ch in codes)
        return "".join(str(MOVES.index(Move((-dy,dx)))) for dx,dy in values)
    moves=rotate_moves(row["moves"])
    proof=certify(level,moves)
    assert proof==row["proof"] and geometry(level.grid)==row["geometry"]
    return dict(row,name=name,source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),
                moves=moves,proof=proof,prefix=rotate_moves(row["prefix"]))


def build():
    FOLDER.mkdir(exist_ok=True)
    forbidden={geometry(c.level.grid) for cases in development_sets() for c in cases}
    forbidden.update(i["geometry"] for i in json.loads((DATA/"test-manifest.json").read_text()).values())
    with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
        forbidden.update(r[0] for r in db.execute("SELECT hash FROM geoms"))
    counts={}
    for bank in ("train","control"):
        rows=[]
        for group in range(1,21):
            step=2 if group in PAIRED else 1
            for i in range(0,50,step):
                check_time()
                for attempt in range(100):
                    if group==19:
                        base=prepare(group,i,bank,attempt)
                        batch=[base,rotate_scenario(base,i+1)]
                    else:
                        batch=[prepare(group,j,bank,attempt) for j in range(i,i+step)]
                    signatures={r["geometry"] for r in batch}
                    if signatures.isdisjoint(forbidden): break
                else: raise RuntimeError(f"Повтор геометрии {bank}/{group}/{i}")
                if group in (4,5,6,11):
                    a,b=map(probe_view,batch)
                    if group==11: assert a==b
                    if group in (4,5):
                        assert a.window==b.window and a.position==b.position and a.memory!=b.memory
                        assert (a.fuel,a.health,a.fuel_distance,a.repair_distance)==(b.fuel,b.health,b.fuel_distance,b.repair_distance)
                        assert (a.last_move,a.moved,a.dx,a.dy)==(b.last_move,b.moved,b.dx,b.dy)
                    if group==6: assert a.memory==b.memory and (a.fuel,a.health)!=(b.fuel,b.health)
                forbidden.update(signatures)
                rows.extend(batch)
        pending=FOLDER/f"{bank}.pending"
        with gzip.open(pending,"wt",encoding="utf-8") as stream:
            for r in rows: stream.write(json.dumps(r,separators=(",",":"))+"\n")
        pending.replace(FOLDER/f"{bank}.jsonl.gz")
        counts[bank]=len(rows)
    evidence=dict(verified_episodes=sum(counts.values()),by_bank=counts,groups=20,
                  group_names=NAMES,sha256={b:hashlib.sha256((FOLDER/f"{b}.jsonl.gz").read_bytes()).hexdigest() for b in counts})
    write_text(FOLDER/"verified.json",json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    log("diagnostics_certified",**counts,groups=20)


if __name__=="__main__": build()
