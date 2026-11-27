import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import rows,read,save,digest,level
from ...teacher import Teacher
from .gate_model import GateNet
from .gates import describe
from .local_model import LocalNet,local_image
from .model import HERE,image
from .symmetry import MOVES,ORDER,observation
from .transfer_model import FullModel


def init():
    global base,w,field,local,gates
    torch.set_num_threads(1)
    pack=torch.load(HERE.parent/"runs/field3200/rl.pt",weights_only=True,map_location="cpu")
    base=FullModel(pack);w=pack["linear"].numpy()
    clean=torch.load(HERE.parent/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu")
    field=FullModel(clean).field
    local=LocalNet().eval();gates=GateNet().eval()
    for net,folder,filename in ((local,"local-12000","results.json"),(gates,"gate-3000","result.json")):
        step=read(HERE/folder/filename)["selected"]
        net.load_state_dict(torch.load(HERE/folder/f"s{step}.pt",weights_only=True,map_location="cpu"))


@torch.inference_mode()
def estimates(v):
    features=[];images=[];metas=[];patches=[];lasts=[];route=[]
    summaries={}
    for j in range(8):
        rotated=observation(v,j)
        board,meta=image(rotated)
        patch,_=local_image(rotated)
        z,ok=base.directions(rotated)
        order=np.array(ORDER[j])
        features.append((z@w)[order])
        if j==0:mask=ok
        assert np.array_equal(ok[order],mask)
        assert np.array_equal([destination(rotated.window,(1,1),m) is not None for m in MOVES],ok)
        images.append(board);metas.append(meta);patches.append(patch)
        lasts.append(MOVES.index(rotated.last_move) if rotated.last_move else 8)
        route.append(describe(board,meta,(v.fuel,v.health))[0])
    x,m,p,last,g=[torch.as_tensor(np.stack(a)) for a in (images,metas,patches,lasts,route)]
    stock=torch.tensor([[v.fuel,v.health]]*8)
    urgent=int(v.health-v.repair_distance<v.fuel-v.fuel_distance)
    f,_,fk=field(x,m,stock,last)
    slope,value,lk=local(p,m,stock,last)
    assert torch.equal(fk,lk)
    _,gq=gates(g)
    order=np.array(ORDER);b=np.arange(8)[:,None]
    for n,s in (("field_down",f.sigmoid()[:,:,urgent]),("field_viable",f.sigmoid()[:,:,3]),
                ("local_down",slope.softmax(-1)[:,:,urgent,0]),("local_viable",value.sigmoid()[:,:,0]),
                ("gate_down",gq.sigmoid()[:,:,0]),("gate_viable",gq.sigmoid()[:,:,1])):
        summaries[n]=s.numpy()[b,order].tolist()
    summaries["policy"]=np.stack(features).tolist()
    return summaries,mask


def trial(job):
    row,path=job
    env=Environment(level(row));teacher=Teacher(env.level)
    labels={};choices={};packets={};irreversible=False
    while env.status==Status.RUNNING and env.steps<150 and not (env.refuels and env.repairs):
        v=env.view
        if v not in choices:
            z,ok=base.directions(v);a=int(np.where(ok,z@w,-1e12).argmax())
            choices[v]=(ok,a)
        ok,a=choices[v]
        if not irreversible:
            _,viable=teacher.labels(env)
            if viable.any() and ok.sum()>1:
                s=(v.fuel-v.fuel_distance,v.health-v.repair_distance)
                keys=[]
                if min(s)<=2 and "first_tight" not in labels:keys.append("first_tight")
                if not viable[a]:
                    keys.append("first_loss");irreversible=True
                if keys:
                    scores,mask=estimates(v)
                    assert np.array_equal(mask,ok)
                    assert np.where(ok,np.array(scores["policy"][0]),-1e12).argmax()==a
                    k=2*int(s[1]<s[0])
                    d=(v.fuel_distance,v.repair_distance)[k//2]
                    down=[bool(ok[j] and min(t[destination(v.memory,v.position,m)] for t in teacher.dist[k:k+2])<d)
                          for j,m in enumerate(MOVES)]
                    packets[str(env.steps)]=dict(legal=ok.tolist(),safe=viable.tolist(),down=down,scores=scores,
                        slack=min(s),stage="one" if env.refuels or env.repairs else "before",action=a)
                    labels.update({key:str(env.steps) for key in keys})
        if irreversible:break
        env.step(MOVES[a])
    teacher.remaining.cache_clear()
    save(path,dict(map=row["name"],tier=row["tier"],labels=labels,packets=packets))


def metrics(states,name):
    down0=down8=safe0=safe8=disagree=0;adiff=[];changes=0;set_changes=0
    for case in states:
        x=np.asarray(case["scores"][name]);ok=np.asarray(case["legal"])
        peak=np.max(x[:,ok],1)
        ties=ok[None]&(np.abs(x-peak[:,None])<1e-5)
        raw=np.where(ok,x,-1e12).argmax(1)
        a0=raw[0];avg=int(np.where(ok,x.mean(0),-1e12).argmax())
        changes+=int((raw[1:]!=a0).sum())
        set_changes+=int((ties[1:]!=ties[0]).any(1).sum())
        disagree+=int(avg!=a0)
        down0+=case["down"][a0];down8+=case["down"][avg]
        safe0+=case["safe"][a0];safe8+=case["safe"][avg]
        adiff.append(float(np.abs(x[:,ok]-x[0,ok]).max()))
    return dict(states=len(states),orientation_pairs=7*len(states),changed_argmax=changes,
                changed_best_sets=set_changes,ensemble_changed=disagree,down=[down0,down8],
                viable=[safe0,safe8],median_max_difference=float(np.median(adiff)) if adiff else 0,
                largest_difference=max(adiff,default=0))


def main():
    source=HERE/"large.jsonl"
    bank=rows(source)[200:300]
    cache=HERE.parent/"cache/rotation-probe";cache.mkdir(exist_ok=True)
    cfg=dict(train=digest(source),index=[200,300],source=digest(HERE/"symmetry_probe.py"),
             transform=digest(HERE/"symmetry.py"),base=digest(HERE.parent/"runs/field3200/rl.pt"),
             sample="first fork slack<=2 and first irreversible before both; training prefix <=150 steps")
    if (cache/"config.json").exists():assert read(cache/"config.json")==cfg
    else:save(cache/"config.json",cfg)
    jobs=[(r,cache/(r["name"]+".json")) for r in bank]
    init()
    pending=[p for p in jobs if not p[1].exists()]
    for j,p in enumerate(pending,1):
        trial(p)
        if j%20==0:print("rotations",j,len(pending),flush=True)
    data=[read(p) for _,p in jobs]
    groups={}
    for key in ("first_tight","first_loss"):
        pairs=[(r["packets"][r["labels"][key]],r["tier"]) for r in data if key in r["labels"]]
        for subgroup,states in (("all",[c for c,_ in pairs]),("slack01",[c for c,_ in pairs if c["slack"]<=1]),
                               ("H3_H4",[c for c,t in pairs if t in ("H3","H4")])):
            groups[f"{key}/{subgroup}"]={n:metrics(states,n) for n in (
                "policy","field_down","field_viable","local_down","local_viable","gate_down","gate_viable")}
    save(HERE/"rotation-probe.json",dict(config=cfg,groups=groups,files={p.name:digest(p) for _,p in jobs}))
    print(groups["first_loss/slack01"],flush=True)


if __name__=="__main__":main()
