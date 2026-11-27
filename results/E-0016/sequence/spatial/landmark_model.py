"""Обучение длин между конкретными станциями и перспектив двух порядков обслуживания."""
import torch
from torch import nn

FIRST=(1,1,2,2,3,3,4,4)
SECOND=(3,4,3,4,1,2,1,2)


def mlp(inp,hidden,out):
    return nn.Sequential(nn.Linear(inp,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,out),nn.SiLU())


def budgets(d1,d2,d3,stock,fuel_first):
    first_stock=torch.where(fuel_first,stock[:,None,None,0],stock[:,None,None,1])
    second_stock=torch.where(fuel_first,stock[:,None,None,1],stock[:,None,None,0])
    return torch.stack((first_stock-d1,second_stock-d1-d2,100-d2-d3,100-d3),-1)


class LandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair=mlp(18,64,48)
        self.landmark=mlp(4*48,80,48)
        self.relation=mlp(48*3,96,64)
        self.length=nn.Linear(64,128)
        self.leg=mlp(16+64+8,96,64)
        self.gradient=nn.Linear(64,3)
        self.route=mlp(3*64+9,96,48)
        self.possible=nn.Linear(48,1)
        self.move=mlp(64*3+48*2,96,64)
        self.output=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
        self.register_buffer("lengths",torch.arange(128,dtype=torch.float32))
        self.register_buffer("first",torch.tensor(FIRST))
        self.register_buffer("second",torch.tensor(SECOND))

    def forward(self,pair,action,meta,stock,last):
        batch=len(pair)
        ok=action[:,:,0,0]>.5
        h=self.pair(pair)
        node=self.landmark(torch.cat((h[:,:,0],h[:,:,1:3].amax(2),h[:,:,3:5].amax(2),h[:,:,5]),-1))
        rel=self.relation(torch.cat((h,node[:,:,None,:].expand(-1,-1,6,-1),
                                      node[:,None,:,:].expand(-1,6,-1,-1)),-1))
        raw=self.length(rel)
        logits=(raw+raw.transpose(1,2))*.5
        bins=self.lengths
        minimum=(pair[...,0]*127).round()
        maximum=(pair[...,1]*127).round()
        logits=logits.masked_fill((bins<minimum[...,None])|(bins>maximum[...,None]),-1e4)
        pr=logits.softmax(-1)
        avg=(pr*bins).sum(-1)
        spread=(pr*(bins-avg[...,None]).square()).sum(-1).clamp_min(.01).sqrt()
        ids=torch.arange(8,device=pair.device)[None,:]
        inv=torch.tensor([2,3,0,1,6,7,4,5,8],device=pair.device)
        resources=torch.cat((stock.float()/100,(stock-meta[:,6:]).float().clamp(-15,45)/20),1)
        common=resources[:,None,None,:].expand(-1,8,5,-1)
        motor=torch.stack((last[:,None]==ids,inv[last.long()][:,None]==ids),-1).float()
        lens=torch.stack((avg[:,0,1:]/80,spread[:,0,1:]/30),-1)
        leg=self.leg(torch.cat((action,rel[:,None,0,1:].expand(-1,8,-1,-1),common,
             motor[:,:,None,:].expand(-1,-1,5,-1),lens[:,None].expand(-1,8,-1,-1)),-1))
        sign=self.gradient(leg)
        distribution=sign.softmax(-1)
        first_length=avg[:,None,0,self.first]+1+distribution[:,:,self.first-1,2]-distribution[:,:,self.first-1,0]
        middle=avg[:,self.first,self.second][:,None].expand(-1,8,-1)
        exit_leg=avg[:,self.second,5][:,None].expand(-1,8,-1)
        margins=budgets(first_length,middle,exit_leg,stock,self.first<3)
        sigma=torch.stack((spread[:,0,self.first],spread[:,self.first,self.second],spread[:,self.second,5]),-1)
        orientation=(self.first<3)[None,None,:,None].expand(batch,8,-1,-1)
        parts=(leg[:,:,self.first-1],rel[:,self.first,self.second][:,None].expand(-1,8,-1,-1),
               rel[:,self.second,5][:,None].expand(-1,8,-1,-1),margins.clamp(-80,100)/40,
               sigma[:,None].expand(-1,8,-1,-1)/35,orientation,first_length[:,:,:,None]/80)
        route=self.route(torch.cat(parts,-1))
        possible=self.possible(route).squeeze(-1)
        mix=possible.softmax(2)
        move=self.move(torch.cat((leg[:,:,:2].amax(2),leg[:,:,2:4].amax(2),leg[:,:,4],
                                  route.amax(2),(route*mix[:,:,:,None]).sum(2)),-1))
        context=move.masked_fill(~ok[:,:,None],-1e4).amax(1,keepdim=True).expand_as(move)
        scores=self.output(torch.cat((move,context),-1))
        return scores,possible,logits,sign,ok


def true_routes(distance,sign,stock):
    first=torch.tensor(FIRST,device=distance.device)
    second=torch.tensor(SECOND,device=distance.device)
    d1=distance[:,None,0,first]+sign[:,:,first-1]
    d2=distance[:,first,second][:,None].expand_as(d1)
    d3=distance[:,second,5][:,None].expand_as(d1)
    slack=budgets(d1,d2,d3,stock,first<3)
    return (slack>=0).all(-1) & (slack[:,:,:,1]+d2>0) & (100-d2>0)
