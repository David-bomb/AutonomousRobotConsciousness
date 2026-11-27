// Conditional road sampling from current visible cells, service measurements and a
// finite set of possible trip origins inferred from the cell memory itself.
#include "belief_worlds.cpp"

namespace {
struct Tunnel {int cell,other;};
using Roads=vector<array<int,8>>;
using Fields=array<vector<int>,5>;

struct Sampler {
    int w,h,n,plazas=0;
    const char* memory;
    vector<char> base;
    vector<vector<Tunnel>> graph;
    vector<Edge> candidates;
    Sampler(const char* m,int w,int h,const int* s):w(w),h(h),n(w*h),memory(m),base(n) {
        for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
            int p=y*w+x;
            if(m[p]!='?'){
                base[p]=m[p]!='#';
                plazas+=base[p]&&!(x&1)&&!(y&1);
            }else if(p==s[0] || ((x&1)&&(y&1)))base[p]=1;
            else if((x&1)!=(y&1)){
                int z=(x&1)?w:1;candidates.push_back({p,p-z,p+z});
            }
        }
        Components groups(n);
        for(int u=w;u<n-w;u++)if(base[u]){
            if(base[u-1])groups.join(u,u-1);
            if(base[u-w])groups.join(u,u-w);
        }
        vector<int> id(n,-1);int count=0;
        for(int u=0;u<n;u++)if(base[u]&&id[groups.root(u)]<0)id[groups.root(u)]=count++;
        graph.resize(count);
        for(auto e:candidates)if(base[e.a]&&base[e.b]){
            int a=id[groups.root(e.a)],b=id[groups.root(e.b)];
            if(a!=b){graph[a].push_back({e.wall,b});graph[b].push_back({e.wall,a});}
        }
    }
    Roads sample(int family,mt19937& rng)const{
        auto cells=base;
        int count=int(graph.size());
        vector<char> used(count);vector<int> stack;
        int first=rng()%count;
        for(int c=0;c<count;c++){
            int root=(first+c)%count;if(used[root])continue;
            stack.push_back(root);used[root]=1;
            while(!stack.empty()){
                int j=family==1?int(rng()%stack.size()):int(stack.size())-1;
                int u=stack[j],choices=0,idx=-1;
                for(int k=0;k<int(graph[u].size());k++)if(!used[graph[u][k].other] && rng()%(++choices)==0)idx=k;
                if(idx<0){stack[j]=stack.back();stack.pop_back();continue;}
                auto e=graph[u][idx];cells[e.cell]=1;used[e.other]=1;stack.push_back(e.other);
            }
        }
        if(family==2){
            vector<int> extra;
            for(int u=w+1;u<n-w-1;u++)if(u%w>0&&u%w<w-1&&memory[u]=='?'&&!cells[u]&&
                ((cells[u-1]&&cells[u+1])||(cells[u-w]&&cells[u+w])))extra.push_back(u);
            shuffle(extra.begin(),extra.end(),rng);
            for(int i=0;i<min(n/30,int(extra.size()));i++)cells[extra[i]]=1;
        }
        if(family==3){
            vector<int> centers;
            for(int y=3;y<h-3;y+=4)for(int x=3;x<w-3;x+=4)centers.push_back(y*w+x);
            shuffle(centers.begin(),centers.end(),rng);int accepted=0;
            for(int u:centers){
                int rad=1+rng()%2;bool ok=true;
                for(int y=-1;y<=1;y++)for(int x=-rad;x<=rad;x++)ok&=memory[u+y*w+x]!='#';
                if(!ok)continue;
                for(int y=-1;y<=1;y++)for(int x=-rad;x<=rad;x++)cells[u+y*w+x]=1;
                if(++accepted>=n/110)break;
            }
        }
        Roads g(n);
        for(int u=0;u<n;u++){
            g[u].fill(-1);if(!cells[u])continue;
            for(int a=0;a<8;a++){
                int x=u+dx[a],y=u+w*dy[a],q=x+w*dy[a];
                if(cells[x] && cells[y] && cells[q])g[u][a]=q;
            }
        }
        return g;
    }
};

double outside(int v,int low,int high){return max(0,max(low-v,v-high));}
struct Tier {int low,high,direct,m1lo,m1hi,m2lo,m2hi;};
const Tier tiers[]={{9,13,28,5,8,7,12},{11,14,38,3,6,5,9},{13,17,48,2,4,3,6},{15,18,58,0,2,1,4}};

double witness(int w,int h,const int* sites,const Fields& d,int fuel,int hp,const int* starts,int count,bool first){
    if(!count)return 0.;
    int small=min(fuel,hp),gap=abs(fuel-hp),k=fuel<=hp?1:3,other=4-k;
    double best=70.;
    for(auto tier:tiers){
        if(outside(w/2,tier.low,tier.high) || outside(h/2,tier.low,tier.high))continue;
        for(int s=0;s<count;s++){
            int a=starts[3*s],lower=starts[3*s+1],geodesic=starts[3*s+2];
            int direct=d[0][a];
            if(direct>=inf)continue;
            double shape=.35*outside(direct,tier.direct,tier.direct+26);
            for(int i=k;i<k+2;i++){
                int early=d[i][a];
                if(early>=inf)continue;
                double approach=shape+.40*outside(early,4,min(27,direct/2))+
                    .50*max(0,direct+2-early-d[0][sites[i]]);
                for(int j=other;j<other+2;j++){
                    int middle=d[i][sites[j]],tail=d[0][sites[j]];
                    if(max(middle,tail)>=inf)continue;
                    double mission=approach+.33*outside(middle,4,43)+.33*max(0,middle+tail-99)
                        +.25*max(0,tail-93)+.3*max(0,early+middle+tier.m2lo+1-direct);
                    for(int margin=tier.m1lo;margin<=tier.m1hi;margin++){
                        int elapsed=early+margin-small;
                        double time=first?.8*abs(elapsed):.75*max(0,lower-elapsed)+.22*max(0,elapsed-lower);
                        double cost=mission+time+.16*(lower-geodesic)+
                            .58*outside(gap+margin-middle,tier.m2lo,tier.m2hi);
                        best=min(best,cost);
                    }
                }
            }
        }
    }
    return best;
}
}

extern "C" __declspec(dllexport) void origin_votes(const char* memory,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int fuel,int hp,const int* starts,int count,int first,
    int particles,double* output){
    Sampler sampler(memory,w,h,sites);
    mt19937 rng(921701);
    vector<array<double,3>> weights(particles);
    vector<array<double,32>> values(particles);
    for(int k=0;k<particles;k++){
        int family=k%4;
        auto g=sampler.sample(family,rng);
        Fields d;for(int i=0;i<5;i++)d[i]=distances(g,sites[i]);
        auto nearest=[&](int kind,int u){return min(d[1+2*kind][u],d[2+2*kind][u]);};
        int urgent=hp-now[1]<fuel-now[0];
        double mismatch=sampler.plazas&&family<2?min(7,sampler.plazas):0;
        for(int i=0;i<2;i++){
            int z=nearest(i,pos);
            mismatch+=(i==urgent?.55:.27)*abs(z-now[i]);
            if(old[i]>=0)mismatch+=(i==urgent?2.6:1.)*abs((nearest(i,prior)-z)-(old[i]-now[i]));
        }
        double cost=witness(w,h,sites,d,fuel,hp,starts,count,first!=0);
        weights[k]={-mismatch,-mismatch-.45*cost,-mismatch-cost};
        for(int a=0;a<8;a++)if(g[pos][a]>=0){
            int q=g[pos][a];const bool at_f=q==sites[1]||q==sites[2],at_r=q==sites[3]||q==sites[4];
            values[k][4*a]=nearest(0,q)<nearest(0,pos);
            values[k][4*a+1]=nearest(1,q)<nearest(1,pos);
            int f=at_f?100:fuel-1,hp2=at_r?100:hp-1;
            int tail=q==sites[0]?0:(min(f,hp2)>0?route(q,f,hp2,sites,d):inf);
            values[k][4*a+2]=tail<inf;
            values[k][4*a+3]=tail<inf?exp(-tail/90.):0.;
        }
    }
    fill(output,output+96,0.);
    for(int arm=0;arm<3;arm++){
        double peak=-1.e20,mass=0.;
        for(auto v:weights)peak=max(peak,v[arm]);
        for(int k=0;k<particles;k++){
            double p=exp(weights[k][arm]-peak);mass+=p;
            for(int j=0;j<32;j++)output[32*arm+j]+=p*values[k][j];
        }
        for(int j=0;j<32;j++)output[32*arm+j]/=mass;
    }
}
