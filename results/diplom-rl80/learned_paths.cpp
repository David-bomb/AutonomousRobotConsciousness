// Estimate finite service geodesics using aggregate transition and local-road counts.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using namespace std;
namespace {
constexpr int dx[]={0,1,0,-1,1,1,-1,-1},dy[]={-1,0,1,0,-1,1,1,-1};
constexpr int opposite[]={2,3,0,1,6,7,4,5},inf=10000,bins=4*6*4*6,stat_width=8*9;
using Edges=vector<array<int,8>>;
struct Context {int bin;array<int,8> canonical;};
Context context(int x,int y,int tx,int ty,int distance){
    int xx=tx-x,yy=ty-y,ax=abs(xx),ay=abs(yy);
    bool swapxy=ax<ay;int big=max(ax,ay),small=min(ax,ay);
    int length=big<=1?0:big<=3?1:big<=6?2:big<=10?3:big<=17?4:5;
    int shape=small==0?0:small*3<big?1:small*3<2*big?2:3;
    int d=distance-big,excess=d<=0?0:d==1?1:d<=3?2:d<=7?3:d<=14?4:5;
    int parity=swapxy?((y&1)+2*(x&1)):((x&1)+2*(y&1));
    Context c{((parity*6+length)*4+shape)*6+excess,{}};
    for(int a=0;a<8;a++){
        int ux=dx[a]*(xx<0?-1:1),uy=dy[a]*(yy<0?-1:1);
        if(swapxy)swap(ux,uy);
        for(int j=0;j<8;j++)if(dx[j]==ux&&dy[j]==uy){c.canonical[a]=j;break;}
    }
    return c;
}
Edges board(const char* m,int w,int h,bool optimistic,bool tree,int goal){
    int n=w*h;Edges g(n);vector<char> road(n);
    for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
        int p=y*w+x;
        road[p]=(m[p]!='#' && (m[p]!='?'||p==goal||optimistic));
        if(tree && !(x&1)&&!(y&1))road[p]=0;
    }
    for(int u=0;u<n;u++){
        g[u].fill(-1);if(!road[u])continue;
        for(int a=0;a<(tree?4:8);a++){
            int x=u+dx[a],y=u+w*dy[a],q=x+w*dy[a];
            if(road[x]&&road[y]&&road[q])g[u][a]=q;
        }
    }
    return g;
}
vector<int> lengths(const Edges& g,int p){
    vector<int> d(g.size(),inf),q{p};d[p]=0;
    for(size_t i=0;i<q.size();i++)for(int v:g[q[i]])if(v>=0&&d[v]==inf){d[v]=d[q[i]]+1;q.push_back(v);}
    return d;
}
}
#ifdef PATH_TRAIN
extern "C" __declspec(dllexport) void learn_paths(const char* m,int w,int h,const int* sites,float* table,float* prior){
    auto g=board(m,w,h,false,false,sites[0]);
    for(int j=0;j<5;j++){
        int s=sites[j];auto d=lengths(g,s);
        for(int p=0;p<w*h;p++){
            if(d[p]<1||d[p]>55)continue;
            auto c=context(p%w,p/w,s%w,s/w,d[p]);int count=0,ways=0,directions=0;
            for(int a=0;a<8;a++)if(g[p][a]>=0){
                count+=d[g[p][a]]==d[p]-1;ways+=d[g[p][a]]==d[p]+1;directions|=1<<c.canonical[a];
            }
            if(!count)continue;
            prior[c.bin*56+d[p]]+=1;
            for(int a=0;a<8;a++)if(g[p][a]>=0&&d[g[p][a]]==d[p]-1){
                auto add=[&](int prev,double increment){
                    float* row=table+(c.bin*9+prev)*stat_width+c.canonical[a]*9;row[0]+=float(increment);
                    for(int z=0;z<8;z++)if(directions&(1<<z))row[z+1]+=float(increment);
                };
                add(8,1./count);
                for(int b=0;b<8;b++)if(g[p][b]>=0&&d[g[p][b]]==d[p]+1)
                    add(c.canonical[opposite[b]],1./(count*ways));
            }
        }
    }
}
#else
namespace {
struct Inference {
    int w,n,pos,previous,d,total_kind,target;
    const int* sites;const int* now;const int* old;const float* table;
    const Edges &known,&allowed;
    const vector<int> &prefix,&preceding;
    const array<vector<int>,5>& forced;
    vector<int> lower;
    vector<double> memo;vector<char> checked;bool motif;
    Inference(int w,int pos,int previous,int kind,int target,const int* sites,const int* now,const int* old,const float* table,
        const Edges& known,const Edges& allowed,const vector<int>& prefix,const vector<int>& preceding,
        const array<vector<int>,5>& roads,bool motif):w(w),n(int(known.size())),pos(pos),previous(previous),d(now[kind]),
        total_kind(kind),target(target),sites(sites),now(now),old(old),table(table),known(known),allowed(allowed),
        prefix(prefix),preceding(preceding),forced(roads),lower(lengths(allowed,sites[target])),
        memo((d+1)*n*9),checked(memo.size()),motif(motif){}
    bool possible(int u,int left)const{
        int taken=d-left;
        if(lower[u]>left||forced[target][u]<left||prefix[u]<taken)return false;
        if(old[total_kind]>=0&&preceding[u]+left<old[total_kind])return false;
        for(int s=1;s<5;s++){
            int k=(s-1)/2;
            if(taken+forced[s][u]<now[k]|| (old[k]>=0&&preceding[u]+forced[s][u]<old[k]))return false;
        }
        return true;
    }
    array<double,8> probabilities(int u,int heading,int left)const{
        int s=sites[target];auto c=context(u%w,u/w,s%w,s/w,left);
        int h=heading==8?8:c.canonical[heading];
        const float* fine=table+(9*c.bin+h)*stat_width;
        const float* coarse=table+(9*c.bin+8)*stat_width;
        double nf=0,nc=0;for(int a=0;a<8;a++){nf+=fine[a*9];nc+=coarse[a*9];}
        double regular=std::min(1.,25./max(nc,1.));
        if(nf<1.)regular=1.;
        array<double,8> pr{};double mass=0;
        for(int a=0;a<8;a++){
            if(allowed[u][a]<0||(heading!=8&&a==opposite[heading]))continue;
            const float* f=fine+9*c.canonical[a];const float* b=coarse+9*c.canonical[a];
            double hits=f[0]+regular*b[0],weight=hits+.06;
            if(motif && hits>0){
                double logscore=0.;
                for(int z=0;z<8;z++)if(z!=a && (known[u][z]>=0||allowed[u][z]<0)){
                    int j=c.canonical[z]+1;
                    double p=(f[j]+regular*b[j]+1.)/(hits+2.);
                    logscore+=log(known[u][z]>=0?p:1.-p);
                }
                weight*=exp(.37*max(logscore,-12.));
            }
            pr[a]=weight;mass+=weight;
        }
        if(mass)for(auto& value:pr)value/=mass;
        return pr;
    }
    double solve(int u,int heading,int left){
        if(!possible(u,left))return 0.;
        if(u==sites[target]||!left)return u==sites[target]&&!left;
        int key=(left*n+u)*9+heading;if(checked[key])return memo[key];
        checked[key]=1;auto prior=probabilities(u,heading,left);double sum=0.;
        for(int a=0;a<8;a++)if(prior[a])sum+=prior[a]*solve(allowed[u][a],a,left-1);
        return memo[key]=sum;
    }
};
}
extern "C" __declspec(dllexport) void infer_paths(const char* m,int w,int h,int pos,int previous,const int* sites,
    const int* now,const int* old,int kind,const float* table,const float* prior,double* out){
    fill(out,out+8*6,0.);int d=now[kind];if(d<=0||d>55)return;
    auto known=board(m,w,h,false,false,sites[0]);
    auto prefix=lengths(known,pos),prev=lengths(known,previous);
    array<vector<int>,5> forced;for(int s=0;s<5;s++)forced[s]=lengths(known,sites[s]);
    bool tree=true;for(int y=2;y<h-1;y+=2)for(int x=2;x<w-1;x+=2){
        int u=y*w+x;if((m[u]!='?'&&m[u]!='#')||u==sites[0])tree=false;
    }
    for(int mode=0;mode<3;mode++){
        if(mode==2&&!tree)continue;auto g=board(m,w,h,true,mode==2,sites[0]);
        for(int s=1+2*kind;s<3+2*kind;s++){
            int goal=sites[s];auto c=context(pos%w,pos/w,goal%w,goal/w,d);
            int base=c.bin/6;double all=0.;
            for(int ex=0;ex<6;ex++)for(int z=1;z<56;z++)all+=prior[(base*6+ex)*56+z];
            double likelihood=(prior[c.bin*56+d]+1.)/(all+55.);
            Inference f(w,pos,previous,kind,s,sites,now,old,table,known,g,prefix,prev,forced,mode!=0);
            if(!f.possible(pos,d))continue;
            auto pr=f.probabilities(pos,8,d);
            for(int a=0;a<8;a++)if(pr[a]>0. && known[pos][a]>=0){
                double z=pr[a]*f.solve(known[pos][a],a,d-1);
                out[6*a+2*mode]+=z;out[6*a+2*mode+1]+=z*likelihood;
            }
        }
    }
    for(int j=0;j<6;j++){
        double sum=0.;for(int a=0;a<8;a++)sum+=out[a*6+j];
        if(sum>0)for(int a=0;a<8;a++)out[a*6+j]/=sum;
    }
}
#endif
