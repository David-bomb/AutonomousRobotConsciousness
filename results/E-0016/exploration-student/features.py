"""Из текущего Observation в признаки ветвей и наблюдаемых границ."""
import ctypes as ct
from pathlib import Path

import numpy as np
from maze.interface import Move, Observation
from neural_rl.perception import immediate

MOVES = tuple(Move)
XY = np.asarray([m.value for m in MOVES],np.float32)
DLL = ct.CDLL(str(Path(__file__).with_name("fronts.dll")))
DLL.describe.argtypes = [ct.c_char_p,*([ct.c_int]*6),np.ctypeslib.ndpointer(np.float32),np.ctypeslib.ndpointer(np.float32)]
DLL.describe.restype = None


def encode(v: Observation):
    x,y = v.position
    w,h = len(v.memory[0]),len(v.memory)
    goal=(y+v.dy)*w+x+v.dx
    raw=np.empty((8,20),np.float32)
    candidates=np.empty((8,8,12),np.float32)
    DLL.describe("".join(v.memory).encode("ascii"),w,h,y*w+x,goal,v.fuel,v.health,raw,candidates)
    legal=immediate(v.window)
    assert np.array_equal(raw[:,12]>0,legal)
    mask=candidates[:,:,0]>0
    fdist=512 if v.fuel_distance is None else v.fuel_distance
    rdist=512 if v.repair_distance is None else v.repair_distance
    swap=(v.health,rdist)<(v.fuel,fdist)
    stocks=np.asarray([v.health,v.fuel] if swap else [v.fuel,v.health],np.float32)
    sensors=np.asarray([rdist,fdist] if swap else [fdist,rdist],np.float32)
    lengths=raw[:,:12].reshape(8,3,4)
    if swap:
        lengths=lengths[:,[0,2,1]]
        candidates=candidates[:,:,[0,1,2,3,5,4,7,6,8,9,11,10]]
    available=np.asarray([stocks.min(),*stocks],np.float32)
    relative=lengths-lengths[legal].min(axis=0)
    # Масштаб и различия между физическими направлениями задаются для каждой цели.
    pieces=[np.minimum(lengths,128).reshape(8,-1)/48,
            np.clip(relative,0,16).reshape(8,-1)/4,
            (lengths[:,:,0]<512),np.clip((available-lengths[:,:,0])/10,-2,2),
            np.clip((available-lengths[:,:,1])/10,-2,2),
            np.clip((lengths[:,1:,1]-sensors)/4,-3,3),
            np.clip((lengths[:,1:,2]-sensors)/8,-3,3),
            (lengths[:,1:,0]==sensors),(lengths[:,1:,1]<=sensors)]
    services=raw[:,[17,16]] if swap else raw[:,16:18]
    last=np.array(v.last_move.value if v.last_move else (0,0))
    goal_norm=max(abs(v.dx),abs(v.dy))
    extras=np.column_stack([np.log1p(raw[:,12])/5,np.log1p(raw[:,13])/4,
                            raw[:,14]/5,raw[:,15]/5,services,raw[:,18],np.minimum(64,raw[:,19])/20,
                            (XY==last).all(1),(XY==-last).all(1),XY,
                            XY[:,0]*np.sign(v.dx),XY[:,1]*np.sign(v.dy),
                            goal_norm-np.abs(np.array([v.dx,v.dy])-XY).max(1)])
    params=[*np.minimum(sensors,128)/48,*stocks/100,*np.clip((stocks-sensors)/8,-3,3),
            v.dx/40,v.dy/40,goal_norm/40,x/w,y/h,legal.sum()/8]
    pieces.extend((extras,np.broadcast_to(params,(8,len(params)))))
    edge=np.concatenate(pieces,axis=1)
    edge[~legal]=0

    d=candidates[:,:,0]
    b=candidates[:,:,2]
    opt=candidates[:,:,3]
    home=candidates[:,:,4:6]
    both_budget=stocks.min()
    slack=both_budget-d[:,:,None]-home
    # В этих величинах только путь по памяти до границы и обратно. Частный
    # случай первого пополнения с точно нулевым остатком отмечен отдельно.
    affordable=np.stack(((d+home[:,:,0]<=stocks[0])&(d+home[:,:,0]<stocks[1]),
                         (d+home[:,:,1]<stocks[0])&(d+home[:,:,1]<=stocks[1])),axis=2)
    route=d+b
    best=np.min(route[mask]) if mask.any() else 0.
    opt_route=d+opt
    best_opt=np.min(opt_route[mask]) if mask.any() else 0.
    tokens=np.concatenate((np.minimum(d,128)[:,:,None]/40,candidates[:,:,1:2]/8,
          np.minimum(candidates[:,:,2:8],128)/48,candidates[:,:,8:10]/32,
          np.minimum(candidates[:,:,10:12],128)/48,
          np.clip((goal_norm-b)/10,-3,3)[:,:,None],np.clip((route-best)/6,0,4)[:,:,None],
          np.clip((opt_route-best_opt)/6,0,4)[:,:,None],
          np.clip((both_budget-d)/12,-3,3)[:,:,None],np.clip(slack/12,-3,3),
          affordable,(home<512)),axis=2)
    tokens[~mask]=0
    return edge.astype(np.float16),tokens.astype(np.float16),legal.copy()
