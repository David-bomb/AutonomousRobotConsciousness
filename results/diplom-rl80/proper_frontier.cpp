// The seven goals use the same known-road and supply semantics as the archived
// renewable score model. Keep all optimal first moves, including equal-length paths.
#include "renewable_features.cpp"
#include <memory>
namespace {
struct Choice {int t,at,mask;State s;};
struct Aim {
    double cost=1.e12;
    int mask=0;
    void offer(double c,int first){
        if(!first)return;
        if(c+1.e-6<cost){cost=c;mask=first;}
        else if(abs(c-cost)<1.e-6)mask|=first;
    }
};
int degree_step(int from,int to,int w){
    for(int a=0;a<8;a++)if(to-from==dy[a]*w+dx[a])return a;
    return -1;
}
void shortest(const Graph& g,int pos,int w,vector<int>& ds,vector<int>& first){
    ds.assign(g.size(),far);first.assign(g.size(),0);
    vector<int> q(g.size());q[0]=pos;ds[pos]=0;int end=1;
    for(int i=0;i<end;i++){
        int u=q[i];
        for(auto e:g[u]){
            int v=e.v, d=ds[u]+1;
            int m=u==pos?(1<<degree_step(u,v,w)):first[u];
            if(ds[v]==far){q[end++]=v;ds[v]=d;first[v]=m;}
            else if(ds[v]==d)first[v]|=m;
        }
    }
}

// All Pareto-optimal first station visits, including later returns with improved stocks.
vector<Choice> choices(const Surface& b,State s,const vector<int>& ds,const vector<int>& first){
    vector<Choice> q{{0,-1,0,s}};
    array<vector<Choice>,5> accepted;
    for(size_t k=0;k<q.size();k++){
        Choice current=q[k];
        for(int j=1;j<5;j++){
            int target=b.sites[j],d=current.at<0?ds[target]:b.kd[current.at][target];
            if(!d || d>=far)continue;
            Choice next{current.t+d,j,current.at<0?first[target]:current.mask,current.s};
            if(!arrive(next.s,d,type(j)))continue;
            auto &group=accepted[j];bool add=true;
            for(auto& v:group){
                if(v.t==next.t && v.s.fuel==next.s.fuel && v.s.hp==next.s.hp){
                    int extra=next.mask&~v.mask;
                    v.mask|=next.mask;
                    if(extra){next.mask=extra;q.push_back(next);}
                    add=false;break;
                }
                if(v.t<=next.t && v.s.fuel>=next.s.fuel && v.s.hp>=next.s.hp){add=false;break;}
            }
            if(add){
                group.erase(remove_if(group.begin(),group.end(),[&](const Choice& v){
                    return next.t<=v.t && next.s.fuel>=v.s.fuel && next.s.hp>=v.s.hp;
                }),group.end());
                group.push_back(next);q.push_back(next);
            }
        }
    }
    vector<Choice> out{q.front()};
    for(const auto& g:accepted)out.insert(out.end(),g.begin(),g.end());
    return out;
}
}
extern "C" __declspec(dllexport) void proper(const char* mem,int w,int h,int pos,const int* sites,int f,int hp,int* masks){
    static unique_ptr<Surface> board;
    if(!board || board->memory!=mem || board->w!=w || board->h!=h || !equal(sites,sites+5,board->sites.begin()))
        board=make_unique<Surface>(mem,w,h,sites);
    auto& b=*board;
    fill(masks,masks+7,0);
    bool basin=false;
    for(int i=1;i<5;i++)basin|=(b.kd[i][pos]<far && b.need[i]<=100);
    if(!basin && b.kd[0][pos]>=far)return;
    vector<int> ds,first;shortest(b.known,pos,w,ds,first);
    auto options=choices(b,{f,hp},ds,first);
    array<Aim,7> goals;
    for(auto l:options){
        const auto& roads=l.at<0?ds:b.kd[l.at];
        int d=roads[sites[0]];
        if(d<=min(l.s.hp,l.s.fuel))goals[0].offer(l.t+d,l.at<0?first[sites[0]]:l.mask);
    }
    if(goals[0].mask){masks[0]=goals[0].mask;return;}
    if(!basin)return;
    for(int stage=0;stage<2;stage++){
        for(auto l:options){
            const auto &roads=l.at<0?ds:b.kd[l.at];
            for(int v:b.fringe){
                int d=roads[v],t=l.t+d, mask=l.at<0?first[v]:l.mask;
                State left{l.s.fuel-d,l.s.hp-d};
                if(!mask || min(left.hp,left.fuel)<=0)continue;
                double discounted_gain=.06*b.gain[v],remaining=b.fd[0][v];
                if(!stage){
                    bool safe=false;
                    for(int i=1;i<5 && !safe;i++){
                        State ret=left;
                        safe=b.need[i]<=100 && arrive(ret,b.kd[i][v],type(i)) &&
                             (type(i)==1?ret.hp:ret.fuel)>=b.need[i];
                    }
                    if(safe){
                        goals[1].offer(t+.02*remaining-discounted_gain,mask);
                        goals[2].offer(t+remaining-discounted_gain,mask);
                        goals[3].offer(t+.35*remaining-discounted_gain,mask);
                    }
                }else{
                    double tail=b.expected(v,left);
                    goals[4].offer(t+tail-discounted_gain,mask);
                    goals[5].offer(t+.45*tail-discounted_gain,mask);
                    goals[6].offer(t+tail+.4*remaining-discounted_gain,mask);
                }
            }
        }
        if(goals[1].mask)break;
    }
    for(int i=1;i<7;i++)masks[i]=goals[i].mask;
}
