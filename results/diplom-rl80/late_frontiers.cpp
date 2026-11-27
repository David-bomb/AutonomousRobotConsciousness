// Route lengths from the ordinary cell memory to an observation frontier and onward.
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <vector>
using namespace std;
namespace {
constexpr int inf=9999;
constexpr int dx[]={0,1,0,-1,1,1,-1,-1}, dy[]={-1,0,1,0,-1,1,1,-1};
struct Road {int q,length;};
using Graph=vector<vector<Road>>;

vector<int> distances(const Graph& roads,int src,int skip=-1,int initial=0){
    vector<int> d(roads.size(),inf);
    using Pair=pair<int,int>;
    priority_queue<Pair,vector<Pair>,greater<Pair>> queue;
    queue.emplace(initial,src);d[src]=initial;
    while(!queue.empty()){
        auto [length,u]=queue.top();queue.pop();
        if(d[u]!=length)continue;
        for(auto [q,w]:roads[u])if(q!=skip && length+w<d[q]){
            d[q]=length+w;queue.emplace(d[q],q);
        }
    }
    return d;
}
struct Stocks{int fuel,health;};
bool arrive(Stocks &s,int dist,int kind){
    if(dist>min(s.fuel,s.health))return false;
    s.fuel-=dist;s.health-=dist;
    if(kind==1)s.fuel=100;
    if(kind==2)s.health=100;
    return kind==0 || min(s.fuel,s.health)>0;
}
}

extern "C" __declspec(dllexport) void frontiers(const char* mem,int w,int h,int pos,const int* sites,int f,int hp,double* output){
    int n=w*h;
    Graph known(n), optimistic(n), cautious(n);
    array<int,8> adjacent;adjacent.fill(-1);
    vector<int> boundaries,gain(n);
    for(int u=0;u<n;u++){
        if(mem[u]=='#')continue;
        int x=u%w,y=u/w;
        for(int m=0;m<8;m++){
            int xx=x+dx[m],yy=y+dy[m];
            if(xx<0||xx>=w||yy<0||yy>=h)continue;
            int q=yy*w+xx,i=y*w+xx,j=yy*w+x;
            gain[u]+=(mem[q]=='?');
            if(mem[q]=='#'||mem[i]=='#'||mem[j]=='#')continue;
            bool hidden=mem[q]=='?'||mem[i]=='?'||mem[j]=='?'||mem[u]=='?';
            if(!hidden){
                known[u].push_back({q,1});
                if(u==pos)adjacent[m]=q;
            }
            optimistic[u].push_back({q,1});
            cautious[u].push_back({q,1+(hidden?1+(dx[m]!=0 && dy[m]!=0):0)});
        }
        if(u!=pos && gain[u] && mem[u]!='?')boundaries.push_back(u);
    }
    array<vector<int>,5> site;
    for(int t=0;t<5;t++)site[t]=distances(known,sites[t]);
    vector<int> lower=distances(optimistic,sites[0]),guessed=distances(cautious,sites[0]);
    fill(output,output+8*9,1.e8);
    auto kind=[](int t){return t<=2?1:2;};
    for(int a=0;a<8;a++){
        int q=adjacent[a];
        if(q<0)continue;
        auto from=distances(known,q,pos,1);
        double* best=output+9*a;
        auto inspect=[&](const vector<int>& paths,int used,Stocks s){
            if(paths[sites[0]]<=min(s.fuel,s.health))best[0]=min(best[0],double(used+paths[sites[0]]));
            for(int u:boundaries){
                int d=paths[u], cost=used+d;
                Stocks left{s.fuel-d,s.health-d};
                if(min(left.fuel,left.health)<=0)continue;
                bool one=false,two=false;
                for(int i=1;i<5;i++){
                    Stocks escaped=left;
                    if(!arrive(escaped,site[i][u],kind(i)))continue;
                    one=true;
                    for(int j=1;j<5;j++)if(kind(i)!=kind(j)){
                        Stocks both=escaped;
                        if(arrive(both,site[i][sites[j]],kind(j)))two=true;
                    }
                }
                double rem=lower[u],forecast=guessed[u],novelty=min(5,gain[u])*.10;
                if(two){
                    best[1]=min(best[1],cost+rem+.03*rem-novelty);
                    best[2]=min(best[2],cost+forecast+.03*forecast-novelty);
                    best[3]=min(best[3],.28*cost+forecast-novelty);
                }
                if(one)best[4]=min(best[4],cost+forecast-novelty);
                int supply=min(left.fuel,left.health);
                if(supply>=rem)best[5]=min(best[5],cost+forecast+.02*rem-novelty);
                if(supply>=forecast)best[6]=min(best[6],cost+forecast+.02*rem-novelty);
                best[7]=min(best[7],cost*.20+forecast+1.3*max(0.,forecast-supply)-novelty);
                best[8]=min(best[8],cost*.28+rem+1.3*max(0.,rem-supply)-novelty);
            }
        };
        inspect(from,0,{f,hp});
        for(int i=1;i<5;i++){
            int d=from[sites[i]];
            Stocks s{f,hp};
            if(!arrive(s,d,kind(i)))continue;
            inspect(site[i],d,s);
            for(int j=1;j<5;j++)if(kind(j)!=kind(i)){
                int ds=site[i][sites[j]];
                Stocks b=s;
                if(arrive(b,ds,kind(j)))inspect(site[j],d+ds,b);
            }
        }
    }
}
