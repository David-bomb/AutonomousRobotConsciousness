"""Map-weighted learning of action-to-action differences at ambiguous early training forks."""
import numpy as np
import torch
import torch.nn.functional as F
from paired_policy import PairHead,Paired,load
from early_trials import compare,physical
from exact_train import balanced
from renewable_train import EXP,HERE,data,digest,save

OUT=HERE/"paired-search"


def corpus(subdir):
    xs=[];ys=[];ms=[];ps=[];ws=[];names=[]
    for p in sorted((HERE/"panel-search"/subdir).glob("*.npz")):
        with np.load(p) as a:
            count=len(a["x"])
            names.append(p.stem)
            if count:
                xs.extend(a["x"]);ys.extend(a["y"]);ms.extend(a["legal"]);ps.extend(a["parent"])
                ws.extend([1/count]*count)
    return (torch.from_numpy(np.array(xs,dtype="f4")),torch.from_numpy(np.array(ys)),torch.from_numpy(np.array(ms)),
            torch.tensor(ps,dtype=torch.long),np.asarray(ws)/sum(ws),names)


def objective(head,x,y,mask):
    c,s=head(x,mask)
    a=y[:,:,None,:];b=y[:,None,:,:]
    active=(mask[:,:,None]&mask[:,None,:])[:,:,:,None] & ~torch.eye(8,dtype=torch.bool)[None,:,:,None]
    diff=a!=b
    target=(a&~b).float()
    # Equal total weight per snapshot irrespective of junction degree.
    change=(F.binary_cross_entropy_with_logits(c,diff.float(),reduction="none")*active).sum((1,2))/active.sum((1,2)).clamp(min=1)
    valid=active&diff
    sign=(F.binary_cross_entropy_with_logits(s,target,reduction="none")*valid).sum((1,2))/valid.sum((1,2)).clamp(min=1)
    return (change+sign)@torch.tensor([1.,.42])


def fit_heads(training,held):
    x,y,m,_,prob,_=training
    xx,yy,mm,base,weights,_=held
    states=[];records=[]
    for seed in (2198721,2198722,2198723):
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        h=PairHead(x.shape[2]);opt=torch.optim.AdamW(h.parameters(),lr=.0016,weight_decay=.02)
        best=float("inf");state=None
        for step in range(1,501):
            idx=rng.choice(len(x),size=96,replace=True,p=prob)
            loss=objective(h,x[idx],y[idx],m[idx]).mean()
            opt.zero_grad();loss.backward();opt.step()
            if step in (80,220,500):
                with torch.inference_mode():
                    logits,sg=h(xx,mm);row=torch.arange(len(xx))
                    gain=(logits[row,:,base,0].sigmoid()*torch.tanh(sg[row,:,base,0]/2)).masked_fill(~mm,-5.)
                    choice=gain.argmax(1)
                    wins=yy[row,choice,0];prior=yy[row,base,0]
                    avg=float(objective(h,xx,yy,mm).double()@torch.from_numpy(weights))
                    records.append(dict(seed=seed,step=step,held_loss=avg,wins=int((wins&~prior).sum()),
                                        losses=int((prior&~wins).sum()),base=int(prior.sum()),n=len(xx)))
                    if avg<best:
                        best=avg;state={k:v.clone() for k,v in h.state_dict().items()}
        states.append(state)
    save(OUT/"supervision.json",dict(fit_states=len(x),held_states=len(xx),checkpoints=records))
    return states


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    current=HERE/"protected-search/fixed/model.pt";old=HERE/"panel-search/model.pt";large=EXP/"sequence/spatial/large.jsonl"
    assert digest(current)=="92b9b74fbd924161e79bf37b7e7b3dadfbafcd0bc2cd7fb0bbdcaee3d18665b1"
    train=corpus("sensor-fit");held=corpus("sensor-check");bank=data.rows(large)
    fit=balanced(bank[8100:8200],15);selection=balanced(bank[9600:9700],25)
    names=set(train[-1]+held[-1])
    assert not names&{r["name"] for r in fit+selection}
    eye=np.eye(12);choices=np.vstack((np.zeros(12),eye,.65*eye[2]+.65*eye[8],.65*eye[6]+.65*eye[11]))
    save(OUT/"protocol.json",dict(parent=digest(current),panel=digest(old),panel_protocol=digest(HERE/"panel-search/protocol.json"),
        fit_maps=train[-1],held_maps=held[-1],fit_returns=[r["name"] for r in fit],select_returns=[r["name"] for r in selection],
        seeds=[2198721,2198722,2198723],options=choices.tolist(),bank=digest(large),
        hashes={n:digest(HERE/n) for n in ("paired_policy.py","paired_train.py","panel_policy.py","early_trials.py")},
        rule="Learn equality and conditional sign for viability and urgent descent for every legal action pair, with uniform map weight. At deployment compare against the actual protected-parent action, penalize inter-seed disagreement. Select each seed checkpoint by map-weighted held likelihood. Full terminal selection: 15 starts per tier followed by 25 disjoint starts per tier; top four nonzero settings and mean against parent, ties to parent. No historical outcomes or held dev labels used."))
    pack=torch.load(current,weights_only=True,map_location="cpu");panel=torch.load(old,weights_only=True,map_location="cpu")
    for key in ("variants","local","field"):pack[key]=panel[key]
    pack.update(format="paired-viability-1",pair_heads=fit_heads(train,held),pair_width=train[0].shape[2],paired=torch.zeros(12,dtype=torch.float64))
    net=Paired(pack);a=compare(OUT,"fit",fit,choices,net)
    leaders=choices[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((choices[0],leaders,leaders.mean(0)))
    b=compare(OUT,"select",selection,pool,net);theta=pool[b.argmax()]
    path=OUT/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp");torch.save(dict(pack,paired=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(path)
    save(OUT/"frozen.json",dict(model_sha256=digest(path),fit=a.tolist(),select=b.tolist(),coefficients=theta.tolist(),
                               protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(path.read_bytes(),ablate))


if __name__=="__main__":main()
