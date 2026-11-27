"""Отбор силы обученной поправки за терминальный SUCCESS на train."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from .common import HERE,SEED,digest,group,read,write
from .fit import freeze,loadnet
from .route_search import run_pool
from .terminal import init_worker


def main():
    torch.set_num_threads(1)
    path=HERE/"weights/post-teacher.pt";net=loadnet(path)
    earlier=read(HERE/"terminal/route-search.json")
    used=set(earlier["first"])|set(earlier["second"])
    held=set(earlier["held_continuations"])
    bank=group("extra")
    cases=[]
    for r in bank[:100]:
        trace=read(HERE/"takeovers"/(r["name"]+".json"))
        if trace["reached"] and r["name"] not in held:cases.append((r,trace))
    full=[(r,None) for r in bank[100:] if r["name"] not in used]
    assert len(cases)==40 and len(full)>=40
    rng=np.random.default_rng(SEED+325);rng.shuffle(cases);rng.shuffle(full)
    scales=[0,.25,.5,.75,1,1.5,2.5,4];options=[]
    for scale in scales:
        w=net.mix.detach().numpy().copy();w[-1]=scale;options.append(w)
    with ProcessPoolExecutor(2,initializer=init_worker,initargs=(path,)) as pool:
        a,first=run_pool(pool,cases[:20]+full[:16],options)
        shortlist=sorted({0,*sorted(range(len(scales)),key=lambda i:-first[i])[:3]})
        b,second=run_pool(pool,cases[20:]+full[16:40],[options[i] for i in shortlist])
    winner=shortlist[max(range(len(shortlist)),key=lambda i:(second[i],first[shortlist[i]],-shortlist[i]))]
    with torch.no_grad():net.mix[-1]=scales[winner]
    log=HERE/"terminal/post-teacher-search.json"
    write(log,dict(scales=scales,first=a,first_wins=first,shortlist=shortlist,second=b,second_wins=second,
                   selected=winner,scale=scales[winner],seed=SEED+325))
    freeze(net,"post-teacher-rl",dict(parent="post-teacher",objective="physical SUCCESS",search_sha256=digest(log),
          scale=scales[winner]))
    print("terminal post-teacher",scales[winner],first,second,flush=True)


if __name__=="__main__":main()
