// Estimate late action values on alternate, sensor-compatible completions of the revealed map.
#include "origin_worlds.cpp"
#include <memory>
#include <string>
namespace {
struct Visit{int site,time,fuel,hp;};
int best_finish(int q,int fuel,int hp,const int* sites,const Fields& d){
    if(q==sites[0])return 0;
    if(min(fuel,hp)<=0)return inf;
    vector<Visit> todo{{-1,0,fuel,hp}};
    array<vector<Visit>,5> admitted;
    int best=inf;
    for(size_t n=0;n<todo.size();n++){
        auto cur=todo[n];
        int p=cur.site<0?q:sites[cur.site],budget=min(cur.fuel,cur.hp);
        if(cur.time>=best)continue;
        if(d[0][p]<=budget)best=min(best,cur.time+d[0][p]);
        for(int i=1;i<5;i++){
            int len=d[i][p];if(!len||len>budget || cur.time+len>=best)continue;
            Visit dest{i,cur.time+len,i<=2?100:cur.fuel-len,i>=3?100:cur.hp-len};
            if(min(dest.fuel,dest.hp)<=0)continue;
            auto& group=admitted[i];
            if(any_of(group.begin(),group.end(),[&](auto a){
                return a.time<=dest.time&&a.hp>=dest.hp&&a.fuel>=dest.fuel;}))continue;
            group.erase(remove_if(group.begin(),group.end(),[&](auto a){
                return a.time>=dest.time&&a.hp<=dest.hp&&a.fuel<=dest.fuel;}),group.end());
            group.push_back(dest);todo.push_back(dest);
        }
    }
    return best;
}
struct Portfolio{
    string memory;int w,h;array<int,5> sites;vector<Fields> worlds;
    Portfolio(const char* map,int w,int h,const int* ids,int samples):memory(map),w(w),h(h){
        copy(ids,ids+5,sites.begin());
        Sampler sampler(map,w,h,ids);mt19937 rng(815402);
        for(int i=0;i<samples;i++){
            auto g=sampler.sample(i%4,rng);Fields state;
            for(int k=0;k<5;k++)state[k]=distances(g,ids[k]);
            worlds.push_back(move(state));
        }
    }
};
}
extern "C" __declspec(dllexport) void latent_finish(const char* map,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int f,int hp,int samples,double* out){
    static unique_ptr<Portfolio> bank;
    if(!bank||bank->memory!=map||bank->w!=w||bank->h!=h||int(bank->worlds.size())!=samples
       || !equal(sites,sites+5,bank->sites.begin()))bank=make_unique<Portfolio>(map,w,h,sites,samples);
    vector<array<double,8*5>> values(samples);
    vector<double> logw(samples);
    for(int i=0;i<samples;i++){
        auto& d=bank->worlds[i];double err=0;
        for(int k=0;k<2;k++){
            int a=1+2*k,b=a+1;
            int reading=min(d[a][pos],d[b][pos]);
            err+=.37*abs(reading-now[k]);
            if(old[k]>=0)err+=1.4*abs((min(d[a][prior],d[b][prior])-reading)-(old[k]-now[k]));
        }
        logw[i]=-err;
        array<int,8> cost;cost.fill(inf);int best=inf;
        for(int a=0;a<8;a++){
            int x=pos+dx[a],y=pos+w*dy[a],q=x+w*dy[a];
            if(map[x]=='#'||map[y]=='#'||map[q]=='#'||map[q]=='?'||map[x]=='?'||map[y]=='?')continue;
            int ff=map[q]=='F'?100:f-1,hh=map[q]=='R'?100:hp-1;
            cost[a]=best_finish(q,ff,hh,sites,d);best=min(best,cost[a]);
        }
        if(best>=inf)continue;
        int ties=int(count(cost.begin(),cost.end(),best));
        for(int a=0;a<8;a++)if(cost[a]<inf){
            auto* v=&values[i][a*5];
            v[0]=1.;
            v[1]=cost[a]==best?1./ties:0.;
            v[2]=exp(-double(cost[a]-best)/2.);
            v[3]=exp(-double(cost[a])/55.);
            v[4]=1./(1.+cost[a]-best);
        }
    }
    fill(out,out+8*5,0.);
    double maxlog=*max_element(logw.begin(),logw.end()),mass=0.;
    for(int i=0;i<samples;i++){
        double wt=exp(logw[i]-maxlog);mass+=wt;
        for(int j=0;j<8*5;j++)out[j]+=wt*values[i][j];
    }
    for(int j=0;j<8*5;j++)out[j]/=mass;
}
