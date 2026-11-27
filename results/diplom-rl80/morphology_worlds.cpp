// Complete an unknown coarse spanning tree before adding rooms or sparse extra openings.
// Neither the true family nor the actual seed is an input.
#include "origin_worlds.cpp"
namespace {
struct Midpoint{int cell,a,b;};
struct Morphology {
    int w,h,n;
    vector<char> observed,central;
    vector<Midpoint> edges;
    vector<int> rooms,plazas;
    Morphology(const char* m,int w,int h,const int* sites):w(w),h(h),n(w*h),observed(m,m+n),central(n){
        observed[sites[0]]='B';
        for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
            int u=y*w+x;
            if((x&1)&&(y&1))central[u]=1;
            else if((x&1)!=(y&1)){
                int k=(x&1)?w:1;
                edges.push_back({u,u-k,u+k});
            }else if(road(u))plazas.push_back(u);
        }
        for(int y=3;y<h-3;y+=4)for(int x=3;x<w-3;x+=4)rooms.push_back(y*w+x);
    }
    bool road(int u)const{return observed[u]!='?'&&observed[u]!='#';}
    bool clear_room(int p,int rad)const{
        for(int j=-1;j<=1;j++)for(int i=-rad;i<=rad;i++)if(observed[p+j*w+i]=='#')return false;
        return true;
    }
    bool room_cover(mt19937& rng,vector<char>& cover)const{
        vector<char> necessary(n);
        for(int p:plazas){
            int x=p%w,y=p/w;
            int cx=(x%4==0?x-1:x+1),cy=(y%4==0?y-1:y+1);
            if(cx<3||cx>=w-3||cy<3||cy>=h-3)return false;
            necessary[cy*w+cx]=1;
        }
        vector<int> selected,candidates;
        for(int p:rooms){
            if(!clear_room(p,1)){if(necessary[p])return false;continue;}
            (necessary[p]?selected:candidates).push_back(p);
        }
        int quota=min(int(rooms.size()),n/110),extra=quota-int(selected.size());
        if(extra<0||extra>int(candidates.size()))return false;
        shuffle(candidates.begin(),candidates.end(),rng);
        selected.insert(selected.end(),candidates.begin(),candidates.begin()+extra);
        for(int p:selected){
            int radius=(rng()%2 && clear_room(p,2))?2:1;
            for(int j=-1;j<=1;j++)for(int i=-radius;i<=radius;i++)cover[p+w*j+i]=1;
        }
        return true;
    }
    bool sample(int family,mt19937& rng,Roads& roads)const{
        vector<char> cells=central,covered(n),mandatory(n);
        if(family<2&&!plazas.empty())return false;
        if(family==3&&!room_cover(rng,covered))return false;
        if(family==2){
            if(int(plazas.size())>n/30)return false;
            for(int p:plazas){
                bool horizontal=observed[p-1]!='#' && observed[p+1]!='#';
                bool vertical=observed[p-w]!='#' && observed[p+w]!='#';
                if(!horizontal&&!vertical)return false;
                int k=(horizontal&&(!vertical || rng()%2))?1:w;
                mandatory[p-k]=mandatory[p+k]=1;
            }
        }
        vector<int> preferred;
        Components sets(n);
        for(int i=0;i<int(edges.size());i++){
            auto e=edges[i];
            if(mandatory[e.cell] || (family!=2 && road(e.cell) && !covered[e.cell])){
                if(observed[e.cell]=='#'||!sets.join(e.a,e.b))return false;
                cells[e.cell]=1;
            }else if(family==2 && road(e.cell))preferred.push_back(i);
        }
        shuffle(preferred.begin(),preferred.end(),rng);
        for(int i:preferred){
            auto e=edges[i];
            if(rng()%100<88 && sets.join(e.a,e.b))cells[e.cell]=1;
        }
        vector<int> idx(n,-1);int components=0;
        for(int p=0;p<n;p++)if(central[p]&&idx[sets.root(p)]<0)idx[sets.root(p)]=components++;
        vector<vector<Tunnel>> graph(components);
        for(auto e:edges)if(observed[e.cell]!='#'){
            int a=idx[sets.root(e.a)],b=idx[sets.root(e.b)];
            if(a!=b){graph[a].push_back({e.cell,b});graph[b].push_back({e.cell,a});}
        }
        vector<char> seen(components);vector<int> stack{int(rng()%components)};seen[stack[0]]=1;int visited=1;
        while(!stack.empty()){
            int j=family==1?int(rng()%stack.size()):int(stack.size())-1;
            int u=stack[j],options=0,chosen=-1;
            for(int i=0;i<int(graph[u].size());i++)if(!seen[graph[u][i].other] && rng()%(++options)==0)chosen=i;
            if(chosen<0){stack[j]=stack.back();stack.pop_back();continue;}
            auto e=graph[u][chosen];cells[e.cell]=1;seen[e.other]=1;visited++;stack.push_back(e.other);
        }
        if(visited!=components)return false;
        if(family==2){
            vector<int> need,optional;
            for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++){
                int p=y*w+x;if(cells[p]||observed[p]=='#')continue;
                if(!((cells[p-1]&&cells[p+1])||(cells[p-w]&&cells[p+w]))){
                    if(road(p))return false;
                    continue;
                }
                (road(p)?need:optional).push_back(p);
            }
            int quota=n/30,more=quota-int(need.size());if(more<0 || more>int(optional.size()))return false;
            shuffle(optional.begin(),optional.end(),rng);
            for(int p:need)cells[p]=1;
            for(int i=0;i<more;i++)cells[optional[i]]=1;
        }
        for(int u=0;u<n;u++){
            if(covered[u])cells[u]=1;
            if((road(u)&&!cells[u])||(observed[u]=='#'&&cells[u]))return false;
        }
        roads.resize(n);
        for(int u=0;u<n;u++){
            roads[u].fill(-1);if(!cells[u])continue;
            for(int a=0;a<8;a++){
                int x=u+dx[a],y=u+w*dy[a],q=x+w*dy[a];
                if(cells[x]&&cells[y]&&cells[q])roads[u][a]=q;
            }
        }
        return true;
    }
};
}
extern "C" __declspec(dllexport) void morph_votes(const char* m,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int fuel,int hp,const int* starts,int num_starts,int initial,
    int particles,double* out,double* stats){
    fill(out,out+6*8*3,0.);
    fill(stats,stats+4*3,0.);
    Morphology sampler(m,w,h,sites);mt19937 rng(778193);
    vector<array<double,24>> features;
    vector<array<double,3>> weights;
    vector<int> families;
    for(int k=0;k<particles;k++){
        Roads g;int f=k%4;if(!sampler.sample(f,rng,g))continue;
        stats[3*f]+=1;
        Fields d;for(int j=0;j<5;j++)d[j]=distances(g,sites[j]);
        auto dist=[&](int kind,int p){return min(d[1+2*kind][p],d[2+2*kind][p]);};
        double err=0.;int totaldelta=0,absolute=0;
        int urgent=hp-now[1]<fuel-now[0];
        for(int i=0;i<2;i++){
            int residual=abs(dist(i,pos)-now[i]),delta=old[i]<0?0:abs((dist(i,prior)-dist(i,pos))-(old[i]-now[i]));
            err+=(i==urgent?.65:.30)*residual+(i==urgent?3.:1.5)*delta;
            totaldelta+=delta;absolute+=residual;
        }
        stats[3*f+1]+=(totaldelta==0);
        stats[3*f+2]+=(totaldelta==0&&absolute==0);
        double cost=witness(w,h,sites,d,fuel,hp,starts,num_starts,initial!=0);
        weights.push_back({-err,-err-.50*cost,-err-1.25*cost});families.push_back(f);
        array<double,24> v{};
        for(int a=0;a<8;a++)if(g[pos][a]>=0){
            int q=g[pos][a],ff=(q==sites[1]||q==sites[2])?100:fuel-1,hh=(q==sites[3]||q==sites[4])?100:hp-1;
            v[3*a]=dist(0,q)<dist(0,pos);v[3*a+1]=dist(1,q)<dist(1,pos);
            v[3*a+2]=q==sites[0]||(min(ff,hh)>0 && route(q,ff,hh,sites,d)<inf);
        }
        features.push_back(v);
    }
    for(int j=0;j<6;j++){
        int arm=j<3?j:1;
        auto take=[&](int f){return j<3 || (j==3&&f<2) || (j==4&&f==2) || (j==5&&f==3);};
        double peak=-1.e20,sum=0.;
        for(size_t k=0;k<features.size();k++)if(take(families[k]))peak=max(peak,weights[k][arm]);
        for(size_t k=0;k<features.size();k++)if(take(families[k])){
            double v=exp(weights[k][arm]-peak);sum+=v;
            for(int i=0;i<24;i++)out[j*24+i]+=v*features[k][i];
        }
        if(sum>0)for(int i=0;i<24;i++)out[j*24+i]/=sum;
    }
}
