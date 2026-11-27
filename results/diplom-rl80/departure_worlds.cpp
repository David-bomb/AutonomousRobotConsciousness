// Estimate the value of committing to the same observable frontier across road hypotheses.
#include "morphology_worlds.cpp"
#include <memory>
#include <string>
namespace {
struct CandidateTerrain {Fields distance; array<vector<int>,5> outside;};
struct Departures {
    string memory;
    int w,h,cap;
    array<int,5> sites;
    vector<char> core;
    vector<CandidateTerrain> parts;
    Departures(const char* m,int w,int h,int pos,const int* ids,int cap):
        memory(m),w(w),h(h),cap(cap),core(w*h) {
        copy(ids,ids+5,sites.begin());
        auto known=[&](int c){return m[c]!='#' && m[c]!='?';};
        vector<int> queue{pos};core[pos]=1;
        for(size_t i=0;i<queue.size();i++){
            int p=queue[i];
            for(int a=0;a<8;a++){
                int x=p+dx[a],y=p+w*dy[a],q=x+w*dy[a];
                if(known(x)&&known(y)&&known(q)&&!core[q]){core[q]=1;queue.push_back(q);}
            }
        }
        Morphology sampler(m,w,h,ids);mt19937 rng(97502039);
        for(int t=0;t<cap*6 && int(parts.size())<cap;t++){
            Roads roads;if(!sampler.sample(t%4,rng,roads))continue;
            CandidateTerrain part;
            for(int s=0;s<5;s++)part.distance[s]=distances(roads,ids[s]);
            // One excursion goes past the observed reachable area before rejoining any known road.
            // Limit the unknown leg to a first goal or remote service. Subsequent legs may use all roads.
            for(int s=0;s<5;s++){
                auto& out=part.outside[s];out.assign(w*h,inf);
                if(core[ids[s]])continue;
                vector<int> q{ids[s]};out[ids[s]]=0;
                for(size_t i=0;i<q.size();i++)for(int v:roads[q[i]])if(v>=0&&out[v]==inf){
                    out[v]=out[q[i]]+1;
                    if(!core[v])q.push_back(v);
                }
            }
            parts.push_back(move(part));
        }
    }
};
}
extern "C" void departure_tails(const char* mem,int w,int h,int pos,int oldpos,const int* sites,
    const int* now,const int* old,int cap,int n,const int* options,double* output){
    static unique_ptr<Departures> cache;
    if(!cache||cache->memory!=mem||cache->w!=w||cache->h!=h||cache->cap!=cap||!cache->core[pos]||
       !equal(sites,sites+5,cache->sites.begin()))cache=make_unique<Departures>(mem,w,h,pos,sites,cap);
    fill(output,output+n*6,0.);
    vector<double> logweights;
    for(auto& part:cache->parts){
        double error=0.;
        for(int k=0;k<2;k++){
            int j=1+2*k;auto nearest=[&](int p){return min(part.distance[j][p],part.distance[j+1][p]);};
            int current=nearest(pos);
            error+=.36*abs(current-now[k]);
            if(old[k]>=0)error+=1.3*abs((nearest(oldpos)-current)-(old[k]-now[k]));
        }
        logweights.push_back(-error);
    }
    if(logweights.empty())return;
    double peak=*max_element(logweights.begin(),logweights.end()),mass=0.;
    for(int i=0;i<int(cache->parts.size());i++){
        auto& p=cache->parts[i];double weight=exp(logweights[i]-peak);mass+=weight;
        for(int k=0;k<n;k++){
            int cell=options[4*k],f=options[4*k+1],h0=options[4*k+2],steps=options[4*k+3];
            constexpr double scales[]={1.,1.35,1.75};
            for(int z=0;z<3;z++){
                int best=inf;
                for(int s=0;s<5;s++){
                    int len=p.outside[s][cell];
                    if(len==inf)continue;
                    int cost=int(ceil(len*scales[z]));
                    if(cost>min(f,h0))continue;
                    if(!s){best=min(best,cost);continue;}
                    int ff=s<3?100:f-cost,hh=s>=3?100:h0-cost;
                    if(min(ff,hh)<=0)continue;
                    int tail=route(sites[s],ff,hh,sites,p.distance);
                    if(tail<inf)best=min(best,cost+tail);
                }
                if(best<inf){
                    output[6*k+z]+=weight;
                    output[6*k+3+z]+=weight*exp(-(best+steps)/75.);
                }
            }
        }
    }
    for(int j=0;j<n*6;j++)output[j]/=mass;
}
