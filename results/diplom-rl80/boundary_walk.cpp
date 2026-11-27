// First-entry distributions from unseen goals into the observed connected component.
#include "proper_frontier.cpp"
namespace {
constexpr int horizon=100;
struct Step {int p,dir;double weight;};
struct Boundary {
    Surface board;
    vector<char> core;
    vector<int> index;
    array<vector<array<float,horizon+1>>,5> counts;
    Boundary(const char* m,int w,int h,int pos,const int* sites):board(m,w,h,sites),core(w*h),index(w*h,-1){
        vector<int> work{pos};core[pos]=1;
        for(size_t i=0;i<work.size();i++)for(auto e:board.known[work[i]])if(!core[e.v]){core[e.v]=1;work.push_back(e.v);}
        for(int j=0;j<int(board.fringe.size());j++)index[board.fringe[j]]=j;
        int n=w*h,plazas=0;
        for(int y=2;y<h-1;y+=2)for(int x=2;x<w-1;x+=2)plazas+=m[y*w+x]!='?'&&m[y*w+x]!='#';
        vector<double> openness(n);
        for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
            int p=w*y+x;
            if(m[p]!='?'){openness[p]=m[p]!='#';continue;}
            if(p==sites[0]||((x&1)&&(y&1)))openness[p]=1.;
            else if((x&1)!=(y&1))openness[p]=.56;
            else openness[p]=plazas?.14:.018;
        }
        vector<vector<Step>> roads(n);
        for(int p=0;p<n;p++)if(openness[p]>0)for(int a=0;a<8;a++){
            int x=p+dx[a],y=p+w*dy[a],q=x+w*dy[a];
            double chance=openness[q];
            if(dx[a]&&dy[a])chance*=openness[x]*openness[y]*.8;
            if(chance>0)roads[p].push_back({q,a,chance});
        }
        constexpr int back[]={2,3,0,1,6,7,4,5};
        vector<vector<Step>> transition(8*n);
        for(int p=0;p<n;p++)if(!core[p])for(int prev=0;prev<8;prev++){
            auto& edges=transition[8*p+prev];double sum=0.;
            for(auto e:roads[p]){
                if(e.dir==back[prev]&&roads[p].size()>1)e.weight*=.025;
                if(e.dir==prev)e.weight*=1.25;
                sum+=e.weight;edges.push_back(e);
            }
            if(sum)for(auto& e:edges)e.weight/=sum;
        }
        for(int s=0;s<5;s++){
            counts[s].resize(board.fringe.size());
            if(core[sites[s]])continue;
            vector<float> mass(n*8),next(n*8);
            vector<int> active,pending;
            for(int a=0;a<8;a++){int idx=8*sites[s]+a;active.push_back(idx);mass[idx]=.125f;}
            for(int t=1;t<=horizon;t++){
                for(int idx:active){
                    double weight=mass[idx];mass[idx]=0.;
                    for(auto e:transition[idx]){
                        float value=float(weight*e.weight);
                        if(value<1.e-13)continue;
                        if(core[e.p]){
                            int j=index[e.p];if(j>=0)counts[s][j][t]+=value;
                        }else{
                            int q=e.p*8+e.dir;
                            if(next[q]==0)pending.push_back(q);
                            next[q]+=value;
                        }
                    }
                }
                mass.swap(next);active.swap(pending);pending.clear();
            }
            for(auto& d:counts[s])for(int t=1;t<=horizon;t++)d[t]+=d[t-1];
        }
    }
};
}
extern "C" __declspec(dllexport) void boundary_values(const char* memory,int w,int h,int pos,const int* sites,
    int fuel,int hp,double* output){
    fill(output,output+8*12,0.);
    static unique_ptr<Boundary> cache;
    if(!cache||cache->board.memory!=memory||cache->board.w!=w||cache->board.h!=h||!cache->core[pos]||
      !equal(sites,sites+5,cache->board.sites.begin()))cache=make_unique<Boundary>(memory,w,h,pos,sites);
    auto& s=*cache;auto& b=s.board;
    vector<int> distance,first;shortest(b.known,pos,w,distance,first);
    auto plans=choices(b,{fuel,hp},distance,first);
    vector<array<double,8*6>> best(b.fringe.size());
    for(auto plan:plans){
        auto& dist=plan.at<0?distance:b.kd[plan.at];
        for(int j=0;j<int(b.fringe.size());j++){
            int cell=b.fringe[j],walk=dist[cell],mask=plan.at<0?first[cell]:plan.mask;
            int f=plan.s.fuel-walk,r=plan.s.hp-walk,budget=min(f,r);
            if(!mask||budget<=0)continue;
            int arrival=walk+plan.t;
            for(int scale=0;scale<3;scale++){
                constexpr double stretch[]={.78,1.12,1.65};
                int t=min(horizon,int(budget*stretch[scale]));
                double direct=s.counts[0][j][t],with_remote=direct;
                for(int k=1;k<5;k++)if(!s.core[sites[k]]){
                    int tail=b.fd[0][sites[k]];
                    // Unknown-road travel time following arrival at a new service has its own cost.
                    int other=k<3?r:f;
                    int reserve=min(85,max(4,int(.42*tail)));
                    int feasible=min(t,int((other-reserve)*stretch[scale]));
                    if(feasible>0)with_remote+=.48*s.counts[k][j][feasible]*exp(-tail/110.);
                }
                for(int a=0;a<8;a++)if(mask&(1<<a)){
                    double cost=exp(-arrival/210.);
                    best[j][a*6+scale]=max(best[j][a*6+scale],direct*cost);
                    best[j][a*6+3+scale]=max(best[j][a*6+3+scale],with_remote*cost);
                }
            }
        }
    }
    for(auto& fringe:best)for(int a=0;a<8;a++)for(int j=0;j<6;j++){
        output[a*12+j]=max(output[a*12+j],fringe[a*6+j]);
        output[a*12+j+6]+=fringe[a*6+j];
    }
    for(int j=0;j<12;j++){
        double top=0.;for(int a=0;a<8;a++)top=max(top,output[a*12+j]);
        if(top>1.e-12)for(int a=0;a<8;a++)output[a*12+j]/=top;
    }
}
