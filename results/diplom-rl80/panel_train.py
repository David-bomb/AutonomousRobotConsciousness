"""Stack independently trained navigation estimates and fit the final action by episode return."""
from importlib import import_module
import json
import numpy as np
import torch
import torch.nn.functional as F
from renewable_train import EXP,HERE,Environment,Status,data,digest,moves,save
from early_trials import compare,physical
from panel_policy import Panel,load
from maze.grid import destination

OUT=HERE/"panel-search"
PARENT=HERE/"service-window/fixed/model.pt"
ALTS=[EXP/"prev-two/weights"/(tag+".pt") for tag in ("early600","early1800")]
FIELD=EXP/"sequence/runs/field3000-clean/rl.pt"
LOCAL=EXP/"sequence/runs/local12000/rl.pt"


def state():
    pack=torch.load(PARENT,weights_only=True,map_location="cpu");variants=[]
    for p in ALTS:
        new=torch.load(p,weights_only=True,map_location="cpu")
        changed={k:v for k,v in new.items() if not torch.equal(v,pack["early"][k])}
        if any(k.startswith("base.") for k in changed):raise ValueError("Early feature bases differ")
        variants.append(changed)
    pack.update(format="learned-service-panel-1",variants=variants,
                field=torch.load(FIELD,weights_only=True,map_location="cpu")["field"],
                local=torch.load(LOCAL,weights_only=True,map_location="cpu")["local"])
    return pack


def independent(all_rows):
    file=OUT/"holdout.jsonl"
    if file.exists():return data.rows(file)
    used={r["geometry"] for r in all_rows}
    used.update(data.excluded())
    for p in (EXP/"data/train.jsonl",EXP/"data/dev.jsonl",EXP/"data/confirmation.jsonl",EXP/"sequence/train.jsonl",
        EXP/"sequence/probe.jsonl",EXP/"sequence/spatial/selection.jsonl",EXP/"sequence/spatial/extra.jsonl",
        HERE/"handover-search/train.jsonl",HERE/"return-search/bank.jsonl"):
        if p.exists():used.update(r["geometry"] for r in data.rows(p))
    group=[]
    for seed in (9165901,9165902):
        for r in data.generate(("panel-train",seed)):
            assert r["geometry"] not in used
            used.add(r["geometry"]);group.append(r)
    tmp=file.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in group),encoding="utf8");tmp.replace(file)
    return group


def collect(tag,maps,net):
    folder=OUT/tag;folder.mkdir(exist_ok=True)
    xs=[];ys=[];masks=[];choices=[];slacks=[]
    Teacher=import_module("results.E-0016.teacher").Teacher
    for i,r in enumerate(maps):
        p=folder/(r["name"]+".npz")
        if not p.exists():
            env=Environment(data.level(r));oracle=Teacher(env.level)
            old=(None,None);xs0=[];ys0=[];masks0=[];choices0=[];slacks0=[];seen=set()
            while env.status==Status.RUNNING and not net.route(env.view,net.cfg):
                v=env.view;key=v,old
                if key in seen:break
                seen.add(key)
                q,x,ok,info=net.features(v,old)
                a=q.argmax()
                if x is not None:
                    k,slack,_,_=info
                    _,viable=oracle.labels(env)
                    delta=[(env._lengths["FR"[k]][destination(v.memory,v.position,m)]<
                           (v.fuel_distance,v.repair_distance)[k]) if ok[j] else False for j,m in enumerate(moves)]
                    xs0.append(x);ys0.append(np.column_stack((viable,delta)));masks0.append(ok)
                    choices0.append(a);slacks0.append(slack)
                old=(v.fuel_distance,v.repair_distance);env.step(moves[a])
            temp=p.with_suffix(".pending")
            with temp.open("wb") as f:np.savez_compressed(f,x=np.asarray(xs0,np.float16),y=np.asarray(ys0,bool),
                    legal=np.asarray(masks0,bool),parent=np.asarray(choices0,np.int8),slack=np.asarray(slacks0,np.int16))
            temp.replace(p)
            oracle.remaining.cache_clear()
        with np.load(p) as saved:
            if len(saved["x"]):
                xs.append(saved["x"]);ys.append(saved["y"]);masks.append(saved["legal"])
                choices.append(saved["parent"]);slacks.append(saved["slack"])
        if (i+1)%50==0:print(tag,i+1,"maps",flush=True)
    return tuple(np.concatenate(col) for col in (xs,ys,masks,choices,slacks))


def fit_sensor(train,held):
    X,Y,M,_,S=train
    x=torch.from_numpy(X).float();y=torch.from_numpy(Y).float();m=torch.from_numpy(M)
    w=torch.zeros((x.shape[2],2),requires_grad=True);b=torch.zeros(2,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=.045);rng=np.random.default_rng(65919)
    for i in range(800):
        ind=rng.integers(len(x),size=192);m0=m[ind];y0=y[ind];z=x[ind]@w+b
        bce=F.binary_cross_entropy_with_logits(z,y0,reduction="none")
        diff=z[:,:,None,:]-z[:,None,:,:]
        pair=(y0[:,:,None,:]>y0[:,None,:,:]) & m0[:,:,None,None] & m0[:,None,:,None]
        rank=(F.softplus(-diff)*pair).sum((1,2))/pair.sum((1,2)).clamp(min=1)
        base=(bce*m0[:,:,None]).sum(1)/m0.sum(1,keepdim=True)
        loss=((base+.6*rank)*(1+.4*torch.tensor(S[ind]<=2)[:,None])).mean()+.001*w.square().sum()
        opt.zero_grad();loss.backward();opt.step()
    weights=w.detach();bias=b.detach()
    X,Y,M,base,S=held;z=X@weights.numpy()+bias.numpy()
    index=np.arange(len(X))
    def report(group):
        answer={"n":int(group.sum())}
        for j,k in enumerate(("viable","descent")):
            choice=np.where(M,z[:,:,j],-1.e6).argmax(1)
            answer[k]=[int(Y[index,base,j][group].sum()),int(Y[index,choice,j][group].sum())]
        return answer
    doc={k:report(idx) for k,idx in (("all",np.ones(len(X),bool)),("tight",S<=2))}
    save(OUT/"sensor.json",dict(train=len(x),held=doc,weights=weights.tolist(),bias=bias.tolist()))
    print("held",doc,flush=True)
    return dict(w=weights,bias=bias)


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    assert digest(PARENT)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    large=EXP/"sequence/spatial/large.jsonl";rows=data.rows(large)
    fresh=independent(rows)
    learned=rows[7000:7400];fit=rows[8100:8160];held=fresh[:100];selection=fresh[100:]
    options=np.array([[0]*7,[1,4,0,3,0,2,0],[1,2,2,4,2,3,0],[1,0,4,7,0,4,0],
        [0,4,0,7,2,4,0],[4,0,6,2,1,2,0],[0,0,4,8,3,2,0],[0,0,0,10,0,4,0],
        [1,3,2,4,4,5,-.4],[0,0,4,4,6,2,-.4],[1,2,0,0,0,0,0]],float)
    save(OUT/"protocol.json",dict(parent=digest(PARENT),inputs={str(p.relative_to(EXP)):digest(p) for p in ALTS+[FIELD,LOCAL]},
        corpus=digest(large),fresh=digest(OUT/"holdout.jsonl"),fit_sensor=[r["name"] for r in learned],
        held_sensor=[r["name"] for r in held],fit_policy=[r["name"] for r in fit],select_policy=[r["name"] for r in selection],
        options=options.tolist(),code={n:digest(HERE/n) for n in ("panel_train.py","panel_policy.py","early_trials.py","geodesic_evidence.dll","tree_paths.dll")},
        method="At early ambiguous service forks pool logits and next-readings from three frozen two-register heads sharing one verified encoder, two different single-snapshot spatial networks, known-road and lattice-tree estimates. Train a shared action classifier for true one-move viability and urgent descent; ten nonzero full-trip updates compete against zero. Select four alternatives and mean vs zero on fresh 100 physical training starts. Freeze and reload before dev. The 12000-map source served the local spatial model previously; both 100-map calibration check and independent terminal selection use new seeds."))
    pack=state();net=Panel(pack)
    a=collect("sensor-fit",learned,net);b=collect("sensor-check",held,net)
    pack["classifier"]=fit_sensor(a,b);net=Panel(pack)
    first=compare(OUT,"fit",fit,options,net)
    top=options[np.argsort(-first[1:],kind="stable")[:4]+1];pool=np.vstack((options[0],top,top.mean(0)))
    second=compare(OUT,"select",selection,pool,net);theta=pool[second.argmax()]
    p=OUT/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp");torch.save(dict(pack,residual=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(p)
    save(OUT/"frozen.json",dict(model_sha256=digest(p),coefficients=theta.tolist(),fit=first.tolist(),select=second.tolist(),
                               protocol_sha256=digest(OUT/"protocol.json")))
    for label,ablate in (("dev",False),("ablated",True)):physical(OUT,label,load(p.read_bytes(),ablate))


if __name__=="__main__":main()
