// Sample latent road completions from an observed map. Inputs contain no hidden cells.
#include "belief_worlds.cpp"
#include <queue>
namespace {
struct Tunnel {int cell,other;};
double witness_cost(int a,int cur,const int* sites,const array<vector<int>,5>& d,
                    int f,int h,int first_move){
    // Before either service, both stocks decrease together. Their difference constrains
    // the separation of the two witness stations under the training distribution.
    bool fuel_first=f<=h;
    int small=min(f,h),gap=abs(f-h);
    double best=20.;
    for(int i=fuel_first?1:3;i<(fuel_first?3:5);i++){
        for(int j=fuel_first?3:1;j<(fuel_first?5:3);j++){
            int middle=d[i][sites[j]],tail=d[0][sites[j]];
            if(max(middle,tail)>=inf)continue;
            double cost=.20*max(0,4-middle)+.12*max(0,middle-43)+.16*max(0,middle+tail-99);
            cost+=.30*max(0,abs(middle+2-gap)-4);
            if(a<0){best=min(best,cost);continue;}
            int early=d[i][a],goal=d[0][a];
            if(max(early,goal)>=inf)continue;
            cost+=.15*max(0,4-early)+.15*max(0,early-27)+.23*max(0,goal+2-early-d[0][sites[i]]);
            cost+=.20*max(0,early+middle+1-goal);
            // At the start the original stocks are known exactly. Later the elapsed
            // time before a first visit is bounded below by a modelled straight-line distance.
            if(first_move)cost+=.48*max(0,early-small)+.33*max(0,small-early-8);
            else {
                int cheb_lower=abs(d[0][a]-d[0][cur]);
                cost+=.36*max(0,small+cheb_lower-early-8);
            }
            best=min(best,cost);
        }
    }
    return best;
}
}
extern "C" __declspec(dllexport) void posterior(const char* grid,int w,int h,int pos,int prior,int origin,
    const int* sites,const int* current,const int* old,int fuel,int hp,int pre_service,int is_start,
    int particles,double* output){
    const int n=w*h;vector<char> filled(n);
    vector<Edge> candidates;
    int known_plaza=0;
    for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
        int p=y*w+x;char ch=grid[p];
        if(ch!='?'){
            filled[p]=ch!='#';
            known_plaza+=!(x&1)&&!(y&1)&&filled[p];
        }else if(p==sites[0] || ((x&1)&&(y&1)))filled[p]=1;
        else if((x&1)!=(y&1)){
            int off=(x&1)?w:1;candidates.push_back({p,p-off,p+off});
        }
    }
    Components coarse(n);
    for(int u=w;u<n-w;u++)if(filled[u]){
        if(filled[u-1])coarse.join(u,u-1);
        if(filled[u-w])coarse.join(u,u-w);
    }
    vector<int> remap(n,-1);int count=0;
    for(int u=0;u<n;u++)if(filled[u]){
        int r=coarse.root(u);
        if(remap[r]<0)remap[r]=count++;
    }
    vector<vector<Tunnel>> g(count);
    for(auto e:candidates)if(filled[e.a]&&filled[e.b]){
        int a=remap[coarse.root(e.a)],b=remap[coarse.root(e.b)];
        if(a!=b){g[a].push_back({e.wall,b});g[b].push_back({e.wall,a});}
    }
    mt19937 rng(921701);
    vector<array<double,48>> preference(particles);
    vector<array<double,3>> importance(particles);
    for(int k=0;k<particles;k++){
        int family=k%4;
        auto world=filled;
        vector<char> used(count);vector<int> active;
        int first=rng()%count;
        for(int c=0;c<count;c++){
            int root=(first+c)%count;
            if(used[root])continue;
            used[root]=1;active.push_back(root);
            while(!active.empty()){
                int j=family==1?int(rng()%active.size()):int(active.size())-1;
                int node=active[j],choices=0,chosen=-1;
                for(int e=0;e<int(g[node].size());e++)if(!used[g[node][e].other]){
                    if(rng()%(++choices)==0)chosen=e;
                }
                if(chosen<0){active[j]=active.back();active.pop_back();continue;}
                Tunnel e=g[node][chosen];world[e.cell]=1;used[e.other]=1;active.push_back(e.other);
            }
        }
        if(family==2){
            vector<int> holes;
            for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
                int u=y*w+x;
                if(grid[u]=='?'&&!world[u]&&((world[u-1]&&world[u+1])||(world[u-w]&&world[u+w])))holes.push_back(u);
            }
            shuffle(holes.begin(),holes.end(),rng);
            for(int j=0;j<min(int(holes.size()),n/30);j++)world[holes[j]]=1;
        }
        if(family==3){
            vector<int> centers;
            for(int y=3;y<h-3;y+=4)for(int x=3;x<w-3;x+=4)centers.push_back(y*w+x);
            shuffle(centers.begin(),centers.end(),rng);int accepted=0;
            for(int p:centers){
                int rad=1+rng()%2;bool ok=true;
                for(int y=-1;y<=1;y++)for(int x=-rad;x<=rad;x++)ok&=(grid[p+y*w+x]!='#');
                if(!ok)continue;
                for(int y=-1;y<=1;y++)for(int x=-rad;x<=rad;x++)world[p+y*w+x]=1;
                if(++accepted>=n/110)break;
            }
        }
        vector<array<int,8>> roads(n);
        for(int u=0;u<n;u++){
            roads[u].fill(-1);if(!world[u])continue;
            for(int i=0;i<8;i++){
                int x=u+dx[i],y=u+w*dy[i],q=x+w*dy[i];
                if(world[x]&&world[y]&&world[q])roads[u][i]=q;
            }
        }
        array<vector<int>,5> ds;
        for(int i=0;i<5;i++)ds[i]=distances(roads,sites[i]);
        auto length=[&](int kind,int u){return min(ds[1+2*kind][u],ds[2+2*kind][u]);};
        int sf=length(0,pos),sr=length(1,pos);
        int urgent=(hp-current[1]<fuel-current[0]);
        double abs_cost=(urgent?.27:.55)*abs(sf-current[0])+(urgent?.55:.27)*abs(sr-current[1]);
        double shape=known_plaza&&family<=1?double(min(6,known_plaza)):0.;
        double delta=0.;
        for(int s=0;s<2;s++)if(old[s]>=0 && prior>=0){
            delta+=(s==urgent?2.6:1.0)*abs((length(s,prior)-length(s,pos))-(old[s]-current[s]));
        }
        double witness=pre_service?witness_cost(origin,pos,sites,ds,fuel,hp,is_start):0.;
        importance[k]={-abs_cost-shape,-abs_cost-shape-delta,-abs_cost-shape-delta-witness};
        int best=inf;array<int,8> trip;trip.fill(inf);
        for(int i=0;i<8;i++){
            int q=roads[pos][i];if(q<0)continue;
            double* r=&preference[k][i*6];
            r[0]=length(0,q)<sf;r[1]=length(1,q)<sr;
            r[2]=ds[0][q]<ds[0][pos];
            int f=q==sites[1]||q==sites[2]?100:fuel-1,hp2=q==sites[3]||q==sites[4]?100:hp-1;
            trip[i]=q==sites[0]?0:(min(f,hp2)>0?route(q,f,hp2,sites,ds):inf);
            r[3]=trip[i]<inf;best=min(best,trip[i]);
        }
        if(best<inf)for(int i=0;i<8;i++)if(trip[i]<inf){
            preference[k][i*6+4]=trip[i]==best;
            preference[k][i*6+5]=1./(1+trip[i]-best);
        }
    }
    fill(output,output+3*48,0.);
    for(int t=0;t<3;t++){
        double peak=-1.e12,mass=0.;
        for(auto weights:importance)peak=max(peak,weights[t]);
        for(int k=0;k<particles;k++){
            double weight=exp(importance[k][t]-peak);mass+=weight;
            for(int j=0;j<48;j++)output[t*48+j]+=weight*preference[k][j];
        }
        for(int j=0;j<48;j++)output[t*48+j]/=mass;
    }
}
