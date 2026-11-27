// Основание — geodesic_evidence.cpp лаборатории diplom-rl80, SHA-256 a1b28d9945daadb9df3aea8ebabafacf864bda4fdbf209b6b05a3e4c532fb4e3.
// Столбцы 9–15 дополнительно требуют совместимости открываемого ребра с обоими датчиками.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using namespace std;
namespace {
constexpr int inf=20000;
constexpr int dx[]={0,1,0,-1,1,1,-1,-1},dy[]={-1,0,1,0,-1,1,1,-1};
using Edges=vector<array<int,8>>;
vector<int> lengths(const Edges& g,int source){
    vector<int> queue(g.size()),dist(g.size(),inf);
    if(source<0)return dist;
    queue[0]=source;dist[source]=0;int end=1;
    for(int k=0;k<end;k++){int u=queue[k];for(int v:g[u])if(v>=0 && dist[v]==inf){
        dist[v]=dist[u]+1;queue[end++]=v;
    }}
    return dist;
}
}
extern "C" __declspec(dllexport) void evidence(const char* grid,int w,int h,int pos,int previous,
    const int* sites,const int* current,const int* prior,double* scores){
    int n=w*h;Edges known(n),optimistic(n);
    for(int u=0;u<n;u++){
        known[u].fill(-1);optimistic[u].fill(-1);
        if(grid[u]=='#')continue;
        int x=u%w,y=u/w;
        for(int a=0;a<8;a++){
            int xx=x+dx[a],yy=y+dy[a];
            if(xx<0||yy<0||xx>=w||yy>=h)continue;
            int v=yy*w+xx,i=y*w+xx,j=yy*w+x;
            if(grid[v]=='#'||grid[i]=='#'||grid[j]=='#')continue;
            optimistic[u][a]=v;
            if(grid[u]!='?'&&grid[v]!='?'&&grid[i]!='?'&&grid[j]!='?')known[u][a]=v;
        }
    }
    auto start=lengths(known,pos),old=lengths(known,previous);
    array<vector<int>,8> first;
    for(int a=0;a<8;a++)first[a]=lengths(known,known[pos][a]);
    fill(scores,scores+8*2*16,0.);
    array<vector<int>,2> service;
    for(int t=0;t<2;t++){
        service[t]=lengths(known,sites[1+2*t]);
        auto second=lengths(known,sites[2+2*t]);
        for(int j=0;j<n;j++)service[t][j]=min(service[t][j],second[j]);
    }
    auto compatible=[&](int u,int b){
        const int ux=u%w,uy=u/w,bx=b%w,by=b/w;
        auto open=[&](int z){return grid[z]!='?' || z==u || z==b || z==uy*w+bx || z==by*w+ux;};
        int from=start[b],before=old[b];array<int,2> to={service[0][b],service[1][b]};
        for(int j=0;j<8;j++){
            int z=optimistic[b][j];
            if(z<0 || !open(z) || !open(by*w+z%w) || !open(z/w*w+bx))continue;
            from=min(from,start[z]+1);before=min(before,old[z]+1);
            for(int t=0;t<2;t++)to[t]=min(to[t],service[t][z]+1);
        }
        for(int t=0;t<2;t++)if(from+to[t]<current[t] || (prior[t]>=0 && before+to[t]<prior[t]))return false;
        return true;
    };
    for(int s=0;s<2;s++){
        int d=current[s],before=prior[s];
        if(d<=0)continue;
        array<vector<int>,2> lower,upper;
        for(int k=0;k<2;k++){
            lower[k]=lengths(optimistic,sites[1+2*s+k]);
            upper[k]=lengths(known,sites[1+2*s+k]);
        }
        for(int a=0;a<8;a++)if(known[pos][a]>=0){
            int q=known[pos][a];double* out=scores+(a*2+s)*16;
            if(min(upper[0][q],upper[1][q])==d-1 || (q==previous && before==d-1)){
                out[0]=1;out[1]=1;out[9]=1;
            }
        }
        for(int u=0;u<n;u++){
            int prefix=start[u];if(prefix>=d)continue;
            if(before>=0 && old[u]+d-prefix<before)continue;
            for(int step=0;step<8;step++){
                int b=optimistic[u][step];
                if(b<0 || known[u][step]>=0 || start[b]<=prefix)continue;
                int remaining=d-prefix-1;
                if(before>=0 && old[b]+remaining<before)continue;
                if(min(upper[0][b],upper[1][b])<remaining)continue;
                bool joint=compatible(u,b);
                // If another definitely open suffix is short, its length bounds
                // all putative shortest paths, irrespective of the selected site.
                for(int site=0;site<2;site++){
                    int optimistic_distance=lower[site][b];
                    if(optimistic_distance>remaining) continue;
                    if(optimistic_distance==0 && remaining!=0)continue;
                    array<int,8> linked{};int count=0;
                    for(int a=0;a<8;a++)if(known[pos][a]>=0 && first[a][u]+1==prefix){
                        int q=known[pos][a];
                        if(before>=0 && old[q]+d-1<before)continue;
                        if(min(upper[0][q],upper[1][q])<d-1)continue;
                        linked[a]=1;++count;
                    }
                    // The 3x3 observation guarantees the very first edge is known.
                    constexpr double stretch[]={1.0,1.5,2.25,3.25,5.0};
                    for(int a=0;a<8;a++)if(linked[a]){
                        double* out=scores+(a*2+s)*16;
                        out[1]=1;out[2]+=1./count;
                        if(joint)out[9]+=1./count;
                        for(int z=0;z<5;z++){
                            double expected=optimistic_distance*stretch[z];
                            double h=2.+.30*expected;
                            double weight=exp(-abs(remaining-expected)/h)/(h*count);
                            out[3+z]+=weight;
                            if(joint)out[10+z]+=weight;
                        }
                        out[8]=max(out[8],double(prefix)/max(1,d));
                        if(joint)out[15]=max(out[15],double(prefix)/max(1,d));
                    }
                }
            }
        }
    }
}
