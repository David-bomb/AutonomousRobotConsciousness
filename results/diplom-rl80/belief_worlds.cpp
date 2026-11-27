// Counterfactual geometry sampled solely from the revealed memory and a fitted morphology prior.
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

namespace {
constexpr int dx[]={0,1,0,-1,1,1,-1,-1},dy[]={-1,0,1,0,-1,1,1,-1}, inf=10000;
struct Components {
    vector<int> parent;
    explicit Components(int n):parent(n,-1){}
    int root(int i){return parent[i]<0?i:parent[i]=root(parent[i]);}
    bool join(int a,int b){
        a=root(a);b=root(b);
        if(a==b)return false;
        if(parent[a]>parent[b])swap(a,b);
        parent[a]+=parent[b];parent[b]=a;return true;
    }
};
struct Edge {int wall, a,b;};

vector<int> distances(const vector<array<int,8>>& edges,int origin){
    vector<int> d(edges.size(),inf),q(edges.size());
    int end=1;
    q[0]=origin;d[origin]=0;
    for(int i=0;i<end;i++){
        int u=q[i];
        for(int v:edges[u])if(v>=0 && d[v]==inf){d[v]=d[u]+1;q[end++]=v;}
    }
    return d;
}

// Best modeled length using up to two en-route services. This is a feature of the
// candidate world, never a feasibility certificate for the actual maze.
int route(int p,int f,int h,const int* sites,const array<vector<int>,5>& d){
    int best=d[0][p]<=min(f,h)?d[0][p]:inf;
    for(int a=1;a<5;a++){
        int before=d[a][p];
        if(before<=0 || before>min(f,h))continue;
        int ff=a<=2?100:f-before, hh=a>=3?100:h-before;
        if(min(ff,hh)<=0)continue;
        int direct=d[0][sites[a]];
        if(direct<=min(ff,hh))best=min(best,before+direct);
        for(int b=1;b<5;b++){
            if((a<=2)==(b<=2))continue;
            int middle=d[b][sites[a]];
            if(middle>min(ff,hh))continue;
            int f2=b<=2?100:ff-middle,h2=b>=3?100:hh-middle,tail=d[0][sites[b]];
            if(min(f2,h2)>0 && tail<=min(f2,h2))best=min(best,before+middle+tail);
        }
    }
    return best;
}
}

extern "C" __declspec(dllexport) void worlds(const char* grid,int w,int h,int start,const int* sites,
    int measured_f,int measured_r,int fuel,int health,const double* morphology,int particles,double* out){
    int n=w*h;
    fill(out,out+8*6,0.);
    vector<char> fixed(n);
    vector<Edge> candidates;
    vector<int> plazas;
    for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
        int u=y*w+x;
        if(grid[u]!='?'){fixed[u]=grid[u]!='#';continue;}
        if((x&1)&&(y&1)){fixed[u]=1;continue;}
        if((x&1)!=(y&1)){
            int offset=(x&1)?w:1;
            candidates.push_back({u,u-offset,u+offset});
        }else plazas.push_back(u);
    }
    // The exit position is part of the ordinary observation even when its cell is unknown.
    fixed[sites[0]]=1;
    Components initial(n);
    for(int u=w;u<n-w;u++)if(fixed[u]){
        if(fixed[u-1])initial.join(u,u-1);
        if(fixed[u-w])initial.join(u,u-w);
    }
    mt19937 rng(239017); // common fixed random stream for every actual world and episode
    uniform_real_distribution<double> coin(0.,1.);
    vector<array<double,8*6>> features(particles);
    vector<double> logweights(particles);
    for(int k=0;k<particles;k++){
        auto open=fixed;
        auto components=initial;
        auto shuffled=candidates;
        shuffle(shuffled.begin(),shuffled.end(),rng);
        int style=k%4;
        for(auto e:shuffled){
            if(!open[e.a] || !open[e.b])continue;
            if(components.join(e.a,e.b) || coin(rng)<morphology[2*style])open[e.wall]=1;
        }
        for(int u:plazas)if(coin(rng)<morphology[2*style+1])open[u]=1;
        vector<array<int,8>> edges(n);
        for(int u=0;u<n;u++){
            edges[u].fill(-1);
            if(!open[u])continue;
            for(int m=0;m<8;m++){
                int a=u+dx[m],b=u+w*dy[m],v=a+w*dy[m];
                if(open[a]&&open[b]&&open[v])edges[u][m]=v;
            }
        }
        array<vector<int>,5> d;
        for(int t=0;t<5;t++)d[t]=distances(edges,sites[t]);
        auto F=[&](int u){return min(d[1][u],d[2][u]);};
        auto R=[&](int u){return min(d[3][u],d[4][u]);};
        int actual_f=F(start),actual_r=R(start);
        int diff_f=abs(actual_f-measured_f),diff_r=abs(actual_r-measured_r);
        double importance_f=fuel-measured_f<=health-measured_r?.42:.22;
        double importance_r=.64-importance_f;
        logweights[k]=-importance_f*diff_f-importance_r*diff_r;
        int best=inf;
        array<int,8> lengths;
        lengths.fill(inf);
        for(int m=0;m<8;m++){
            int v=edges[start][m];
            if(v<0)continue;
            int f=(v==sites[1]||v==sites[2])?100:fuel-1;
            int r=(v==sites[3]||v==sites[4])?100:health-1;
            lengths[m]=v==sites[0]?0:(min(f,r)>0?route(v,f,r,sites,d):inf);
            best=min(best,lengths[m]);
            auto* vals=&features[k][m*6];
            vals[0]=F(v)<actual_f;vals[1]=R(v)<actual_r;vals[2]=d[0][v]<d[0][start];
            vals[3]=lengths[m]<inf;
        }
        if(best<inf)for(int m=0;m<8;m++)if(lengths[m]<inf){
            features[k][m*6+4]=lengths[m]==best;
            features[k][m*6+5]=1./(1.+lengths[m]-best);
        }
    }
    double peak=*max_element(logweights.begin(),logweights.end()),total=0.;
    for(int k=0;k<particles;k++){
        double weight=exp(logweights[k]-peak);
        total+=weight;
        for(int a=0;a<8*6;a++)out[a]+=weight*features[k][a];
    }
    for(int a=0;a<8*6;a++)out[a]/=total;
}
