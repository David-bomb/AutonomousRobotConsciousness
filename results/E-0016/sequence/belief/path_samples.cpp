// Условные гипотезы о дороге до сервиса, совместимые с двумя текущими длинами.
// Полная физическая карта в этот расчёт не передаётся.
#include "../geometry.cpp"
#include <random>
#include <cstdint>
using Layer=vector<array<double,8>>;

extern "C" __declspec(dllexport) void paths(const char *grid,int w,int h,int origin,int d,int second,
        char target,const float *road,int trials,float *out){
    fill(out,out+8*5,0.f);
    if(d<=0||d>=100)return;
    const int n=w*h,inv[]={2,3,0,1,6,7,4,5};
    Board observed(grid,grid,w,h,0), possible(grid,grid,w,h,1);
    auto start=observed.distances(origin);
    Dist end(n,inf),lower(n,inf),other(n,inf);
    vector<int> goals,neighbors;
    for(int i=0;i<n;i++){
        if(grid[i]==target)goals.push_back(i);
        if(grid[i]==(target=='F'?'R':'F'))neighbors.push_back(i);
    }
    for(int s:goals){
        auto a=observed.distances(s),b=possible.distances(s);
        for(int i=0;i<n;i++){end[i]=min(end[i],a[i]);lower[i]=min(lower[i],b[i]);}
    }
    for(int s:neighbors){
        auto a=observed.distances(s);for(int i=0;i<n;i++)other[i]=min(other[i],a[i]);
    }
    vector<array<double,8>> edge(n);
    for(int p=0;p<n;p++)for(int a=0;a<8;a++){
        int q=possible.edges[p][a];
        if(q<0)continue;
        double chance=road[q];
        if(a>=4)chance*=road[p/w*w+q%w]*road[q/w*w+p%w];
        edge[p][a]=chance<.002?0:chance*chance;
    }
    vector<Layer> messages(d,Layer(n));
    for(int s:goals)messages[0][s].fill(1);
    for(int rem=1;rem<d;rem++){
        int t=d-rem;
        for(int i=0;i<n;i++){
            if(grid[i]==target||start[i]<t||end[i]<rem||lower[i]>rem||other[i]+t<second)continue;
            double term[8]={};
            for(int a=0;a<8;a++)if(edge[i][a]>0)term[a]=edge[i][a]*messages[rem-1][possible.edges[i][a]][a];
            for(int prev=0;prev<8;prev++)for(int a=0;a<8;a++)if(a!=inv[prev])messages[rem][i][prev]+=term[a];
        }
    }
    uint32_t hash=2166136261u;
    for(int i=0;i<n;i++){hash^=uint8_t(grid[i]);hash*=16777619u;}
    mt19937 rng(hash^(origin*65537u+d*97u+uint8_t(target)));
    vector<char> mandatory(n);
    vector<int> times(n),queue(n),dist(n);
    auto validate=[&](){
        fill(dist.begin(),dist.end(),inf);
        int begin=0,sz=1;queue[0]=origin;dist[origin]=0;
        while(begin<sz){
            int p=queue[begin++],next=dist[p]+1;
            if(next>max(d,second))break;
            for(int a=0;a<8;a++){
                int q=possible.edges[p][a];
                if(q<0||dist[q]!=inf||!mandatory[q])continue;
                if(a>=4 && (!mandatory[p/w*w+q%w]||!mandatory[q/w*w+p%w]))continue;
                if((grid[q]==target&&next<d)||(grid[q]==(target=='F'?'R':'F')&&next<second))return false;
                dist[q]=next;queue[sz++]=q;
            }
        }
        return true;
    };
    double base[8]={},weighted[8]={},sum=0,good=0;
    for(int a=0;a<8;a++){
        int first=observed.edges[origin][a];
        if(first<0)continue;
        base[a]=edge[origin][a]*messages[d-1][first][a];sum+=base[a];
        if(base[a]<=0)continue;
        int complete=0,accepted=0;
        double imp=0.;
        for(int iter=0;iter<trials;iter++){
            for(int i=0;i<n;i++)mandatory[i]=grid[i]!='#'&&grid[i]!='?';
            fill(times.begin(),times.end(),-1);
            int p=first,prev=a;
            times[origin]=0;times[p]=1;
            double ratio=1.;
            bool stopped=false;
            for(int t=1;t<d;t++){
                double all=0,eligible=0,cdf[8];
                for(int next=0;next<8;next++){
                    int q=possible.edges[p][next];
                    double mass=q<0||next==inv[prev]?0:edge[p][next]*messages[d-t-1][q][next];
                    all+=mass;
                    if(q>=0 && times[q]<0)eligible+=mass;
                    cdf[next]=eligible;
                }
                if(eligible<=0){stopped=true;break;}
                ratio*=eligible/all;
                double u=generate_canonical<double,32>(rng)*eligible;
                int next=0;while(next<7&&cdf[next]<=u)next++;
                int q=possible.edges[p][next];mandatory[q]=1;
                if(next>=4){mandatory[p/w*w+q%w]=1;mandatory[q/w*w+p%w]=1;}
                p=q;prev=next;times[p]=t+1;
            }
            if(stopped)continue;
            complete++;
            if(validate()){accepted++;imp+=ratio;}
        }
        out[a*5+2]=float(complete)/trials;out[a*5+3]=float(accepted)/trials;
        out[a*5+4]=float(imp/trials);
        weighted[a]=base[a]*imp/trials;good+=weighted[a];
    }
    for(int a=0;a<8;a++){
        out[a*5]=sum?float(base[a]/sum):0;
        out[a*5+1]=good?float(weighted[a]/good):0;
    }
}
