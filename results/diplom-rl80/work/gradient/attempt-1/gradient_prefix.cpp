// Estimate the first step of a service geodesic from counted prefixes on training maps.
// Inference uses the observation and aggregate counts. Training geometry never survives fit().
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <vector>
using namespace std;

namespace {
constexpr int DX[8]={0,1,0,-1,1,1,-1,-1}, DY[8]={-1,0,1,0,-1,1,1,-1};
constexpr int K=8, MAX_D=48, INF=10000;
using Hist=map<uint32_t,uint32_t>;
map<uint32_t,Hist> counts;

struct Board {
    const char* cells;
    int w,h;
    vector<array<int,8>> edges;
    Board(const char* cells,int w,int h,bool known):cells(cells),w(w),h(h),edges(w*h) {
        auto open=[&](int u){return cells[u]!='#' && (!known || cells[u]!='?');};
        for(int u=0;u<w*h;u++){
            edges[u].fill(-1);
            if(!open(u))continue;
            int x=u%w,y=u/w;
            for(int m=0;m<8;m++){
                int a=x+DX[m], b=y+DY[m];
                if(a>=0 && a<w && b>=0 && b<h && open(b*w+a) && open(y*w+a) && open(b*w+x))
                    edges[u][m]=b*w+a;
            }
        }
    }
    vector<int> lengths(int start) const {
        vector<int> d(w*h,INF);
        queue<int> todo;
        todo.push(start); d[start]=0;
        while(!todo.empty()){
            int u=todo.front(); todo.pop();
            for(int v:edges[u])if(v>=0 && d[v]==INF){d[v]=d[u]+1;todo.push(v);}
        }
        return d;
    }
};

struct Symmetry {
    bool swap;
    int sx,sy;
    int action[8]{}, inverse[8]{};
    Symmetry(bool swap,int sx,int sy):swap(swap),sx(sx),sy(sy) {
        for(int m=0;m<8;m++){
            int a=sx*(swap?DY[m]:DX[m]),b=sy*(swap?DX[m]:DY[m]);
            for(int n=0;n<8;n++)if(DX[n]==a && DY[n]==b){action[m]=n;inverse[n]=m;}
        }
    }
};
vector<Symmetry> transforms(){
    vector<Symmetry> syms;
    for(bool swap:{false,true})for(int sx:{1,-1})for(int sy:{1,-1})syms.emplace_back(swap,sx,sy);
    return syms;
}
const auto syms=transforms();
int parity(int x,int y,bool swap){return swap? (y&1)+2*(x&1):(x&1)+2*(y&1);}
int key(int a,int b,int d,int p){return (((a*65+b)*64+d)*4+p);}
}

extern "C" __declspec(dllexport) void fit(const char* grid,int w,int h,unsigned seed){
    Board real(grid,w,h,true);
    mt19937 rng(seed);
    for(int target=0;target<w*h;target++){
        if(grid[target]!='F' && grid[target]!='R')continue;
        auto dist=real.lengths(target);
        vector<int> chosen(w*h,-1);
        for(int u=0;u<w*h;u++){
            if(dist[u]==0 || dist[u]>MAX_D)continue;
            array<int,8> options;
            int n=0;
            for(int m=0;m<8;m++){
                int v=real.edges[u][m];
                if(v>=0 && dist[v]+1==dist[u])options[n++]=m;
            }
            chosen[u]=options[uniform_int_distribution<int>(0,n-1)(rng)];
        }
        for(int u=0;u<w*h;u++){
            int d=dist[u];
            if(d<1 || d>MAX_D)continue;
            int x=u%w,y=u/w,dx=target%w-x,dy=target/w-y;
            array<int,K> sequence;
            int v=u;
            for(int t=0;t<min(K,d);t++){sequence[t]=chosen[v];v=real.edges[v][chosen[v]];}
            for(const auto& s:syms){
                int a=s.sx*(s.swap?dy:dx),b=s.sy*(s.swap?dx:dy);
                if(a<b || b<0)continue;
                uint32_t prefix=0;
                for(int t=0;t<min(K,d);t++)prefix|=s.action[sequence[t]]<<(3*t);
                counts[key(a,b,d,parity(x,y,s.swap))][prefix]++;
            }
        }
    }
}

// Persist only tuples of conditioning variables, canonical action prefixes and multiplicities.
extern "C" __declspec(dllexport) int save_bank(const char* path){
    ofstream f(path,ios::binary);
    int rows=0;
    for(const auto& bin:counts)for(auto [prefix,n]:bin.second){
        uint32_t data[3]={bin.first,prefix,n};
        f.write(reinterpret_cast<const char*>(data),sizeof data);
        rows++;
    }
    return f?rows:-1;
}
extern "C" __declspec(dllexport) int load_bank(const char* path){
    ifstream f(path,ios::binary);
    if(!f)return -1;
    counts.clear();
    uint32_t data[3];
    int rows=0;
    while(f.read(reinterpret_cast<char*>(data),sizeof data)){counts[data[0]][data[1]]=data[2];rows++;}
    return f.eof() && f.gcount()==0? rows:-1;
}

extern "C" __declspec(dllexport) void predict(const char* grid,int w,int h,int origin,int d,int site_kind,double* out){
    fill(out,out+8,0.);
    if(d<1 || d>MAX_D)return;
    Board possible(grid,w,h,false),known(grid,w,h,true);
    auto from=known.lengths(origin);
    int x=origin%w,y=origin/w;
    for(int target=0;target<w*h;target++){
        if(grid[target]!=site_kind)continue;
        auto lower=possible.lengths(target);
        if(lower[origin]>d)continue;
        auto upper=known.lengths(target);
        int dx=target%w-x,dy=target/w-y;
        for(const auto& s:syms){
            int a=s.sx*(s.swap?dy:dx),b=s.sy*(s.swap?dx:dy);
            if(a<b || b<0)continue;
            // Kernel smoothing across nearby endpoints and lengths with compatible prefixes.
            for(int i=max(0,a-2);i<=a+2;i++)for(int j=max(0,b-2);j<=min(i,b+2);j++){
                for(int z=max(min(K,d),d-2);z<=min(MAX_D,d+2);z++){
                    auto bin=counts.find(key(i,j,z,parity(x,y,s.swap)));
                    if(bin==counts.end())continue;
                    int deviation=abs(i-a)+abs(j-b)+abs(z-d);
                    double factor=pow(.37,deviation);
                    for(auto [prefix,n]:bin->second){
                        int u=origin,first=s.inverse[prefix&7];
                        bool ok=true;
                        for(int t=1;t<=min(K,d);t++){
                            int m=s.inverse[(prefix>>(3*(t-1)))&7];
                            u=possible.edges[u][m];
                            if(u<0 || from[u]<t || lower[u]>d-t || upper[u]<d-t ||
                               (grid[u]==site_kind && t<d)){
                                ok=false;break;
                            }
                        }
                        if(ok)out[first]+=factor*n;
                    }
                }
            }
            break; // Dihedral ties were pooled during fitting.
        }
    }
}
