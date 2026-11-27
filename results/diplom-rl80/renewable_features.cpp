// Action values for a known path to a new observation, with repeatable resupply.
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <queue>
#include <string>
#include <vector>
using namespace std;
namespace {
constexpr int far=20000;
constexpr int dx[]={0,1,0,-1,1,1,-1,-1},dy[]={-1,0,1,0,-1,1,1,-1};
using Dist=vector<int>;
struct Edge {int v,price;};
using Graph=vector<vector<Edge>>;
Dist distances(const Graph &g,int src){
    Dist d(g.size(),far);d[src]=0;
    using Pair=pair<int,int>;
    priority_queue<Pair,vector<Pair>,greater<Pair>> heap;heap.push({0,src});
    while(!heap.empty()){
        auto [cost,u]=heap.top();heap.pop();if(d[u]!=cost) continue;
        for(auto [v,w]:g[u])if(cost+w<d[v]){d[v]=cost+w;heap.push({cost+w,v});}
    }
    return d;
}
struct State {int fuel,hp;};
int type(int i){return i==0?0:(i<=2?1:2);}
bool arrive(State& s,int len,int tag){
    if(len>min(s.fuel,s.hp))return false;
    s.fuel-=len;s.hp-=len;
    if(tag==1)s.fuel=100;
    if(tag==2)s.hp=100;
    return tag==0 || min(s.fuel,s.hp)>0;
}
struct Label{int at,time;State stock;};
bool dominates(const Label&a,const Label&b){
    return a.time<=b.time && a.stock.fuel>=b.stock.fuel && a.stock.hp>=b.stock.hp;
}
struct Surface{
    string memory;int w,h;array<int,5> sites;
    Graph known,forecast;array<Dist,5> kd,fd;
    array<int,5> need;
    vector<int> fringe,gain;
    Surface(string m,int width,int height,const int* ids):
        memory(move(m)),w(width),h(height),known(w*h),forecast(w*h),gain(w*h){
        copy(ids,ids+5,sites.begin());
        for(int u=0;u<w*h;u++){
            if(memory[u]=='#')continue;
            int x=u%w,y=u/w;
            for(int a=0;a<8;a++){
                int xx=x+dx[a],yy=y+dy[a];
                if(xx<=0||yy<=0||xx>=w-1||yy>=h-1)continue;
                int v=yy*w+xx,i=y*w+xx,j=yy*w+x;
                gain[u]+=(memory[v]=='?');
                if(memory[v]=='#'||memory[i]=='#'||memory[j]=='#')continue;
                bool hidden=memory[u]=='?'||memory[v]=='?'||memory[i]=='?'||memory[j]=='?';
                if(!hidden)known[u].push_back({v,1});
                // Unknown edges receive a larger cost than measured open roads.
                forecast[u].push_back({v,hidden?(dx[a]&&dy[a]?4:3):1});
            }
            if(gain[u] && memory[u]!='?') fringe.push_back(u);
        }
        for(int t=0;t<5;t++){kd[t]=distances(known,sites[t]);fd[t]=distances(forecast,sites[t]);}
        need.fill(far);
        for(int i=1;i<5;i++)for(int j=1;j<5;j++)if(type(i)!=type(j)){
            int d=kd[i][sites[j]];
            if(2*d<=100)need[i]=min(need[i],d);
        }
        bool changed=true;
        while(changed){
            changed=false;
            for(int i=1;i<5;i++)for(int j=1;j<5;j++)if(i!=j && need[j]<far){
                int d=kd[i][sites[j]],v=far;
                if(type(i)==type(j))v=d+need[j];
                else if(100-d>=need[j])v=d;
                if(v<=100 && v<need[i]){need[i]=v;changed=true;}
            }
        }
    }
    vector<Label> stops(int start,State initial,const Dist& from)const{
        vector<Label> queue{{-1,0,initial}};
        array<vector<Label>,5> admitted;
        for(size_t cursor=0;cursor<queue.size();cursor++){
            auto s=queue[cursor];
            if(s.at>=0 && any_of(admitted[s.at].begin(),admitted[s.at].end(),[&](const Label& a){
                return dominates(a,s) && (a.time<s.time || a.stock.fuel>s.stock.fuel || a.stock.hp>s.stock.hp);
            }))continue;
            for(int j=1;j<5;j++){
                const int d=s.at<0?from[sites[j]]:kd[s.at][sites[j]];
                if(!d || d>=far)continue;
                Label next{j,s.time+d,s.stock};
                if(!arrive(next.stock,d,type(j)))continue;
                auto& labels=admitted[j];
                if(any_of(labels.begin(),labels.end(),[&](const Label& a){return dominates(a,next);}))continue;
                labels.erase(remove_if(labels.begin(),labels.end(),[&](const Label& a){return dominates(next,a);}),labels.end());
                labels.push_back(next);queue.push_back(next);
            }
        }
        vector<Label> labels{queue.front()};
        for(const auto& group:admitted)labels.insert(labels.end(),group.begin(),group.end());
        return labels;
    }
    double expected(int b,State s)const{
        // Minimum forecast travel time with a penalty for unavailable resources.
        auto penalty=[](int d,State state){return max(0,d-min(state.fuel,state.hp));};
        int direct=fd[0][b];
        double best=direct+3.*penalty(direct,s);
        for(int i=1;i<5;i++){
            int d=fd[i][b],p=penalty(d,s);
            State at{max(0,s.fuel-d),max(0,s.hp-d)};
            if(type(i)==1)at.fuel=100;else at.hp=100;
            int tail=fd[0][sites[i]];
            best=min(best,double(d+tail+3*(p+penalty(tail,at))));
            for(int j=1;j<5;j++)if(type(j)!=type(i)){
                int hop=fd[i][sites[j]],p2=p+penalty(hop,at);
                State last{max(0,at.fuel-hop),max(0,at.hp-hop)};
                if(type(j)==1) last.fuel=100;else last.hp=100;
                int rest=fd[0][sites[j]];
                best=min(best,double(d+hop+rest+3*(p2+penalty(rest,last))));
            }
        }
        return best;
    }
    void scores(int pos,State before,double* out)const{
        fill(out,out+8*7,1.e8);
        for(int a=0;a<8;a++){
            int u=pos+dy[a]*w+dx[a];
            if(none_of(known[pos].begin(),known[pos].end(),[&](Edge e){return e.v==u;}))continue;
            auto* score=out+7*a;
            if(u==sites[0]){score[0]=1;continue;}
            State s=before;
            int tag=memory[u]=='F'?1:(memory[u]=='R'?2:3);
            if(!arrive(s,1,tag))continue;
            Dist from=distances(known,u);
            bool basin=false;
            for(int i=1;i<5;i++)basin|=(from[sites[i]]<far && need[i]<=100);
            for(auto l:stops(u,s,from)){
                const auto& road=l.at<0?from:kd[l.at];
                if(road[sites[0]]<=min(l.stock.fuel,l.stock.hp))
                    score[0]=min(score[0],double(1+l.time+road[sites[0]]));
                if(!basin)continue;
                for(int b:fringe){
                    int d=road[b],t=1+l.time+d;
                    State left{l.stock.fuel-d,l.stock.hp-d};
                    if(min(left.fuel,left.hp)<=0)continue;
                    bool renewable=false;
                    for(int i=1;i<5;i++){
                        State ret=left;
                        if(need[i]<=100 && arrive(ret,kd[i][b],type(i)) &&
                           (type(i)==1?ret.hp:ret.fuel)>=need[i])renewable=true;
                    }
                    const double novelty=gain[b]*.06;
                    const double remaining=fd[0][b];
                    if(renewable){
                        score[1]=min(score[1],t+.02*remaining-novelty);
                        score[2]=min(score[2],t+remaining-novelty);
                        score[3]=min(score[3],t+.35*remaining-novelty);
                    }
                    const double tail=expected(b,left);
                    score[4]=min(score[4],t+tail-novelty);
                    score[5]=min(score[5],t+.45*tail-novelty);
                    score[6]=min(score[6],t+tail+.40*remaining-novelty);
                }
            }
        }
    }
};
}
extern "C" __declspec(dllexport) void renewable(const char* mem,int w,int h,int pos,const int* sites,int f,int hp,double* output){
    // Pure memoization, including all inputs to geometry construction in the cache key.
    static Surface surface(mem,w,h,sites);
    if(surface.w!=w || surface.h!=h || surface.memory!=mem || !equal(sites,sites+5,surface.sites.begin()))
        surface=Surface(mem,w,h,sites);
    surface.scores(pos,{f,hp},output);
}
