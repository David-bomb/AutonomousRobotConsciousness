// Числа продолжений из текущего снимка при заданной измеренной длине.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using namespace std;
namespace {
constexpr int inf=100000;
constexpr int dx[]={0,1,0,-1,1,1,-1,-1},dy[]={-1,0,1,0,-1,1,1,-1};
constexpr int back[]={2,3,0,1,6,7,4,5};
using Graph=vector<array<int,8>>;
vector<int> distances(const Graph& graph,int start){
    vector<int> d(graph.size(),inf),q(graph.size());d[start]=0;q[0]=start;int end=1;
    for(int i=0;i<end;i++)for(int v:graph[q[i]])if(v>=0&&d[v]==inf){
        d[v]=d[q[i]]+1;q[end++]=v;
    }
    return d;
}
}
extern "C" __declspec(dllexport) void path_counts(const char* grid,int w,int h,int pos,int previous,
 const int* ids,const int* current,const int* prior,const double* walls,double* out){
    int n=w*h;
    fill(out,out+8*2*2*6,0.);
    Graph known(n);array<Graph,2> graphs={Graph(n),Graph(n)};
    auto usable=[&](int u,int mode){
        if(grid[u]!='?')return grid[u]!='#';
        if(mode==0)return false;
        if(mode==1)return true;
        int x=u%w,y=u/w,index=(x==0||y==0||x+1==w||y+1==h)?4:2*(y%2)+x%2;
        return walls[index]<.82;
    };
    for(int mode=0;mode<3;mode++){
        Graph& graph=mode?graphs[mode-1]:known;
        for(int u=0;u<n;u++){
            graph[u].fill(-1);if(!usable(u,mode))continue;
            int x=u%w,y=u/w;
            for(int a=0;a<8;a++){
                int xx=x+dx[a],yy=y+dy[a];if(xx<0||yy<0||xx>=w||yy>=h)continue;
                if(usable(yy*w+xx,mode)&&usable(yy*w+x,mode)&&usable(y*w+xx,mode))
                    graph[u][a]=yy*w+xx;
            }
        }
    }
    auto from=distances(known,pos),old=distances(known,previous);
    array<vector<int>,4> upper;
    for(int i=0;i<4;i++)upper[i]=distances(known,ids[i+1]);
    for(int mode=0;mode<2;mode++){
        const auto& g=graphs[mode];
        for(int s=0;s<2;s++){
            int d=current[s],other=1-s;if(d<=0||d>250)continue;
            for(int z=0;z<2;z++){
                int target=ids[1+2*s+z];auto lower=distances(g,target);
                int bridge=min(upper[2*other][target],upper[2*other+1][target]);
                vector<array<double,8>> next(n),now(n),pn(n),pw(n);
                for(int a=0;a<8;a++)next[target][a]=pn[target][a]=1.;
                for(int i=d-1;i>=1;i--){
                    int rem=d-i;
                    for(int u=0;u<n;u++){
                        now[u].fill(0.);pw[u].fill(0.);
                        if(lower[u]>rem || from[u]<i || old[u]+rem<prior[s])continue;
                        if(min(upper[2*s][u],upper[2*s+1][u])<rem)continue;
                        int to_other=min(upper[2*other][u],upper[2*other+1][u]);
                        if(i+to_other<current[other] || i+1+to_other<prior[other])continue;
                        if(from[u]+rem+bridge<current[other] || old[u]+rem+bridge<prior[other])continue;
                        array<double,8> ways{},weighted{};double sum=0.,psum=0.;int degree=0;
                        for(int a=0;a<8;a++)if(int v=g[u][a];v>=0){
                            ways[a]=next[v][back[a]];weighted[a]=pn[v][back[a]];
                            sum+=ways[a];psum+=weighted[a];degree++;
                        }
                        for(int a=0;a<8;a++){
                            now[u][a]=max(0.,sum-ways[a]);
                            pw[u][a]=max(0.,psum-weighted[a])/max(1,degree-1);
                        }
                    }
                    next.swap(now);pn.swap(pw);
                }
                for(int a=0;a<8;a++)if(int u=known[pos][a];u>=0){
                    double *r=out+((a*2+s)*2+mode)*6;
                    double count=next[u][back[a]],prob=pn[u][back[a]];
                    r[z]=log1p(count);r[2+z]=log1p(prob*1.e8);
                    r[4+z]=count>0;
                }
            }
        }
    }
}
