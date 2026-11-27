// Direction likelihood for the lattice-tree component of the training distribution.
// The true maze family is never passed to this calculation.
#include "belief_worlds.cpp"
namespace {
using Four=vector<array<int,4>>;
using Values=array<double,3>;
vector<int> bfs(const Four& g,int start){
    vector<int> d(g.size(),inf),q(g.size());d[start]=0;q[0]=start;int n=1;
    for(int i=0;i<n;i++)for(int v:g[q[i]])if(v>=0 && d[v]==inf){d[v]=d[q[i]]+1;q[n++]=v;}
    return d;
}
struct Exit{int u,v,step,first;};
struct RoadTree {
    int w,h,n,pos,prior;
    Four fixed,allowed;
    vector<int> root,start,previous,first;
    array<vector<int>,5> known,optimistic;
    vector<Exit> exits;
    bool possible=true;
    RoadTree(const char* m,int w,int h,int pos,int prior,const int* sites):
      w(w),h(h),n(w*h),pos(pos),prior(prior),fixed(n),allowed(n),root(n),first(n,-1){
        vector<char> open(n),edge(n);
        for(auto& row:fixed)row.fill(-1);
        for(auto& row:allowed)row.fill(-1);
        for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
            int u=y*w+x;
            bool mark=m[u]!='?'&&m[u]!='#';
            if(u==sites[0])mark=true;
            if((x&1)&&(y&1)){if(m[u]=='#')possible=false;open[u]=1;}
            else if(!(x&1)&&!(y&1)){if(mark)possible=false;}
            else if(mark)open[u]=1;
            else if(m[u]=='?')edge[u]=1;
        }
        if(!possible)return;
        Components sets(n);
        for(int u=w+1;u<n-w;u++)if(open[u]){
            for(int a=0;a<4;a++){
                int v=u+dy[a]*w+dx[a];if(!open[v])continue;
                fixed[u][a]=allowed[u][a]=v;
                if(v<u && !sets.join(u,v))possible=false;
            }
        }
        if(!possible || !open[pos] || !open[prior]){possible=false;return;}
        for(int i=0;i<n;i++)root[i]=sets.root(i);
        for(int u=w+1;u<n-w;u++)if(edge[u]){
            int axis=(u%w)&1?0:1;
            int v=u+dy[axis]*w+dx[axis],z=u-dy[axis]*w-dx[axis];
            if(!open[v]||!open[z]||root[v]==root[z])continue;
            allowed[u][axis]=v;allowed[v][axis^2]=u;
            allowed[u][axis^2]=z;allowed[z][axis]=u;
        }
        start=bfs(fixed,pos);previous=bfs(fixed,prior);
        for(int i=0;i<5;i++){known[i]=bfs(fixed,sites[i]);optimistic[i]=bfs(allowed,sites[i]);}
        vector<int> q(n);q[0]=pos;int tail=1;
        for(int i=0;i<tail;i++){
            int u=q[i];
            for(int a=0;a<4;a++){
                int v=fixed[u][a];
                if(v>=0 && start[v]==start[u]+1){first[v]=u==pos?a:first[u];q[tail++]=v;}
                else if(v<0 && allowed[u][a]>=0)
                    exits.push_back({u,allowed[u][a],a,u==pos?a:first[u]});
            }
        }
    }
};

struct Walk {
    const RoadTree& t;const int* sites;const int* now;
    int kind,target,d;
    vector<Values> memo;
    vector<char> seen;
    Walk(const RoadTree& t,const int* sites,const int* now,int kind,int target,int d):
         t(t),sites(sites),now(now),kind(kind),target(target),d(d),memo((d+1)*4*t.n),seen(memo.size()){}
    Values suffix(int u,int heading,int left){
        Values zero{};
        int sofar=d-left;
        for(int j=1;j<5;j++)if(t.known[j][u]+sofar<now[(j-1)/2])return zero;
        int p=sites[target];
        if(t.optimistic[target][u]>left || ((left-abs(p%t.w-u%t.w)-abs(p/t.w-u/t.w))&1))return zero;
        if(t.root[u]==t.root[p]){
            if(t.known[target][u]!=left)return zero;
            return {1.,1.,1.};
        }
        if(left==0 || t.root[u]==t.root[t.pos])return zero;
        int idx=(left*t.n+u)*4+heading;
        if(seen[idx])return memo[idx];
        seen[idx]=1;
        Values totals{},den{};
        for(int a=0;a<4;a++)if(a!=(heading^2)){
            int v=t.allowed[u][a];if(v<0||t.root[v]==t.root[t.pos])continue;
            // A hidden road beyond an intersection competes with all admissible roads.
            Values w={1.,a==heading?2.5:1.,a==heading?.55:1.};
            auto s=suffix(v,a,left-1);
            for(int i=0;i<3;i++){totals[i]+=w[i]*s[i];den[i]+=w[i];}
        }
        for(int i=0;i<3;i++)if(den[i])totals[i]/=den[i];
        return memo[idx]=totals;
    }
};
}
extern "C" __declspec(dllexport) void tree_evidence(const char* mem,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,double* out){
    fill(out,out+8*2*6,0.);
    RoadTree t(mem,w,h,pos,prior,sites);
    if(!t.possible)return;
    for(int k=0;k<2;k++){
        bool viable=false,was=old[k]<0;
        for(int s=1+2*k;s<3+2*k;s++){
            if(t.known[s][pos]<now[k] || (old[k]>=0 && t.known[s][prior]<old[k]))return;
            auto feasible=[&](int p,int d){
                if(t.optimistic[s][p]>d)return false;
                int v=sites[s];
                if((d-abs(p%w-v%w)-abs(p/w-v/w))&1)return false;
                return t.known[s][p]==inf || t.known[s][p]==d;
            };
            viable|=feasible(pos,now[k]);
            if(old[k]>=0)was|=feasible(prior,old[k]);
        }
        if(!viable || !was)return;
    }
    for(int k=0;k<2;k++){
        int d=now[k];
        if(d<=0 || d>55)continue;
        for(int j=0;j<2;j++){
            int s=1+2*k+j;
            if(t.known[s][pos]==d){
                int a=t.first[sites[s]];if(a<0)continue;
                double* r=out+(a*2+k)*6;
                r[0]=1.;r[1]+=1.;r[2]+=1.;r[3]+=1.;r[4]+=1.;r[5]=1.;
                continue;
            }
            if(t.known[s][pos]<inf || t.optimistic[s][pos]>d)continue;
            Walk walk(t,sites,now,k,s,d);
            for(auto e:t.exits){
                int prefix=t.start[e.u]+1,left=d-prefix;
                if(left<0 || e.first<0)continue;
                int other=1+2*k+(1-j);
                int past=t.previous[e.u]+1+left;
                if(old[k]>=0 && (past<old[k] || (past>old[k] && t.optimistic[other][prior]>old[k])))continue;
                auto mass=walk.suffix(e.v,e.step,left);
                if(mass[0]<=0)continue;
                double* r=out+(e.first*2+k)*6;
                r[1]+=mass[0];r[2]+=mass[1];r[3]+=mass[2];r[4]+=1.;
                r[5]=max(r[5],double(prefix)/d);
            }
        }
        for(int z=1;z<=4;z++){
            double sum=0.;
            for(int a=0;a<4;a++)sum+=out[(a*2+k)*6+z];
            if(sum)for(int a=0;a<4;a++)out[(a*2+k)*6+z]/=sum;
        }
    }
}
