// Two-step partially observed planning in self-simulated, sensor-compatible terrain.
#include "morphology_worlds.cpp"
#include <unordered_map>
namespace {
constexpr int value_count=4;
using Utility=array<double,value_count>;
struct Pending{
    int signature[8];
    array<Utility,8> first{};
    array<array<Utility,8>,8> second{};
    double logweight;
};
struct Totals{
    double mass=0.;
    array<Utility,8> scores{};
};
struct Stock{int f,h;};
Stock pay(Stock s,int p,const int* sites){
    return {(p==sites[1]||p==sites[2])?100:s.f-1,(p==sites[3]||p==sites[4])?100:s.h-1};
}
}
extern "C" __declspec(dllexport) void sensor_lookahead(const char* memory,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int fuel,int hp,const int* starts,int num_starts,int initial,
    int particles,double* output){
    fill(output,output+8*16,0.);
    Morphology model(memory,w,h,sites);mt19937 rng(1036751);
    vector<Pending> samples;
    const int urgent=(hp-now[1]<fuel-now[0]);
    const int other=1-urgent;
    for(int n=0;n<particles;n++){
        Roads g;if(!model.sample(n%4,rng,g))continue;
        Fields d;for(int i=0;i<5;i++)d[i]=distances(g,sites[i]);
        auto nearest=[&](int k,int p){return min(d[k*2+1][p],d[k*2+2][p]);};
        int nf=nearest(0,pos),nr=nearest(1,pos);
        if(max(nf,nr)>=inf)continue;
        double err=0.;
        for(int k=0;k<2;k++){
            int here=nearest(k,pos);
            err+=(k==urgent?.62:.36)*abs(here-now[k]);
            if(old[k]>=0)err+=(k==urgent?2.9:1.5)*abs((nearest(k,prior)-here)-(old[k]-now[k]));
        }
        Pending result{};fill(result.signature,result.signature+8,-1);
        result.logweight=-err-.8*witness(w,h,sites,d,fuel,hp,starts,num_starts,initial!=0);
        // Preserve the observed resource slack when a hypothetical distance
        // differs from reality. Real simulator resources are never modified.
        Stock entry{max(1,min(100,fuel+nf-now[0])),max(1,min(100,hp+nr-now[1]))};
        int initial_path=route(pos,entry.f,entry.h,sites,d);
        auto assess=[&](int p,Stock s,int elapsed,int touched)->Utility{
            if(p==sites[0])return {1.4,1.4,1.4,1.4};
            if(min(s.f,s.h)<=0)return {};
            int stock=urgent?s.h:s.f,nd=nearest(urgent,p);
            bool survives=touched || nd<=stock || d[0][p]<=stock;
            int second_d=nearest(other,p),second_stock=other?s.h:s.f;
            double advance=touched?1.:clamp((nearest(urgent,pos)-nd+elapsed)/(2.*elapsed),0.,1.);
            double secondary=clamp((nearest(other,pos)-second_d+elapsed)/(2.*elapsed),0.,1.);
            int reach=route(p,s.f,s.h,sites,d);
            double progress=initial_path<inf && reach<inf?exp(-max(0,reach+elapsed-initial_path)/3.):0.;
            return {double(survives)+.32*advance+.04*secondary,
                    double(survives)*(1.+.18*advance+.16*(second_d<=second_stock)),
                    double(reach<inf)+.22*advance+.15*progress,
                    .70*survives+.45*advance+.25*progress};
        };
        for(int a=0;a<8;a++){
            int v=g[pos][a];if(v<0)continue;
            auto s=pay(entry,v,sites);
            int seen_first=(v==sites[2*urgent+1]||v==sites[2*urgent+2]);
            result.first[a]=assess(v,s,1,seen_first);
            int legal=0;
            for(int b=0;b<8;b++)if(g[v][b]>=0)legal|=1<<b;
            // Only data exposed by the robot at this prospective next position
            // enter the observation group. Each group shares one next action.
            result.signature[a]=legal+256*((nearest(0,v)-nf+1)+3*(nearest(1,v)-nr+1));
            for(int b=0;b<8;b++)if(g[v][b]>=0){
                if(v==sites[0] || min(s.f,s.h)<=0){result.second[a][b]=result.first[a];continue;}
                int q=g[v][b];
                result.second[a][b]=assess(q,pay(s,q,sites),2,seen_first||q==sites[2*urgent+1]||q==sites[2*urgent+2]);
            }
        }
        samples.push_back(move(result));
    }
    if(samples.empty())return;
    double peak=-1.e20,mass=0.;for(auto& s:samples)peak=max(peak,s.logweight);
    array<unordered_map<int,Totals>,8> groups;array<Totals,8> blind;
    for(auto& s:samples){
        double weight=exp(s.logweight-peak);mass+=weight;
        for(int a=0;a<8;a++)if(s.signature[a]>=0){
            auto& group=groups[a][s.signature[a]];group.mass+=weight;
            for(int j=0;j<4;j++){
                output[a*16+j]+=weight*s.first[a][j];
                for(int b=0;b<8;b++){
                    double weighted=weight*s.second[a][b][j];
                    group.scores[b][j]+=weighted;blind[a].scores[b][j]+=weighted;
                }
            }
        }
    }
    for(int a=0;a<8;a++)for(int j=0;j<4;j++){
        double adaptive=0.,nonadaptive=0.;
        for(auto& item:groups[a]){
            double best=0.;for(int b=0;b<8;b++)best=max(best,item.second.scores[b][j]);
            adaptive+=best;
        }
        for(int b=0;b<8;b++)nonadaptive=max(nonadaptive,blind[a].scores[b][j]);
        output[a*16+j]/=mass;
        output[a*16+4+j]=adaptive/mass;
        output[a*16+8+j]=nonadaptive/mass;
        output[a*16+12+j]=(adaptive-nonadaptive)/mass;
    }
}
