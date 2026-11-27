// Геометрия только наблюдаемой памяти. Никаких правил выбора первого хода.
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

namespace {
using std::vector;
constexpr int inf = 512;
constexpr int dx[] = {0,1,0,-1,1,1,-1,-1};
constexpr int dy[] = {-1,0,1,0,-1,1,1,-1};
struct Edge {int to; bool known;};
using Graph = vector<vector<Edge>>;

vector<int> lengths(const Graph& graph, const vector<int>& starts, int excluded, int mode) {
    vector<int> dist(graph.size(), inf);
    // Малые целые длины. Дополнительные две единицы в гипотезе 2 только на неизвестном ребре.
    vector<vector<int>> levels(inf);
    for (int s:starts) if (s!=excluded) {levels[0].push_back(s); dist[s]=0;}
    int last=0;
    for (int d=0;d<=last;++d) for (int p:levels[d]) {
        if (dist[p]!=d) continue;
        for (const Edge& e:graph[p]) {
            if (e.to==excluded || (!e.known && mode==0)) continue;
            int next=d+1+(mode==2 && !e.known ? 2:0);
            if (next<dist[e.to]) {
                dist[e.to]=next;
                levels[next].push_back(e.to);
                last=std::max(last,next);
            }
        }
    }
    return dist;
}

int cheb(int p,int q,int w) {return std::max(std::abs(p%w-q%w),std::abs(p/w-q/w));}
int nearest(int p,const vector<int>& targets,int w) {
    int d=inf;
    for (int s:targets) d=std::min(d,cheb(p,s,w));
    return d;
}
}

extern "C" __declspec(dllexport) void describe(const char* grid, int w, int h, int pos, int goal,
        int fuel, int health, float* desc, float* fronts) {
    const int n=w*h;
    Graph graph(n);
    std::array<vector<int>,3> sites;
    sites[0].push_back(goal);
    vector<int> gain(n);
    std::array<int,8> neighbors;
    neighbors.fill(-1);
    for (int p=0;p<n;++p) {
        if (grid[p]=='#') continue;
        if (grid[p]=='F') sites[1].push_back(p);
        if (grid[p]=='R') sites[2].push_back(p);
        int x=p%w,y=p/w;
        for (int i=0;i<8;++i) {
            int xx=x+dx[i], yy=y+dy[i];
            if (xx<0 || yy<0 || xx>=w || yy>=h) continue;
            int q=yy*w+xx, u=y*w+xx, v=yy*w+x;
            gain[p] += grid[q]=='?';
            if (grid[q]=='#' || grid[u]=='#' || grid[v]=='#') continue;
            bool known=grid[p]!='?' && grid[q]!='?' && grid[u]!='?' && grid[v]!='?';
            graph[p].push_back({q,known});
            if (p==pos && known) neighbors[i]=q;
        }
    }
    std::array<vector<int>,3> relaxed;
    for (int k=0;k<3;++k) relaxed[k]=lengths(graph,sites[k],-1,1);
    auto to_f=lengths(graph,sites[1],-1,0);
    auto to_r=lengths(graph,sites[2],-1,0);
    std::fill(desc,desc+8*20,0.f);
    std::fill(fronts,fronts+8*8*12,0.f);
    for (int a=0;a<8;++a) {
        int first=neighbors[a];
        if (first<0) continue;
        float* v=desc+a*20;
        auto known=lengths(graph,{first},pos,0);
        auto flat=lengths(graph,{first},pos,1);
        auto weighted=lengths(graph,{first},pos,2);
        for (int k=0;k<3;++k) {
            const vector<int>* fields[]={&known,&flat,&weighted};
            for (int j=0;j<3;++j) {
                int d=inf;
                for (int s:sites[k]) d=std::min(d,1+(*fields[j])[s]);
                v[k*4+j]=float(d);
            }
            v[4*k+3]=float(1+nearest(first,sites[k],w));
        }
        v[14]=float(gain[first]);
        for (auto e:graph[first]) if (e.known && e.to!=pos) ++v[15];
        v[16]=grid[first]=='F'; v[17]=grid[first]=='R'; v[18]=first==goal;
        v[19]=float(inf);
        std::array<int,8> picks;
        std::array<int,8> best;
        picks.fill(-1); best.fill(100000000);
        for (int p=0;p<n;++p) {
            if (known[p]>=inf) continue;
            ++v[12];
            if (gain[p]==0) continue;
            ++v[13];
            int d=known[p]+1,b=cheb(p,goal,w), g=gain[p];
            v[19]=std::min(v[19],float(d));
            // Каждая проекция выделяет представителя множества границ в данной ветви.
            bool back_f=d+to_f[p]<=fuel && d+to_f[p]<health;
            bool back_r=d+to_r[p]<fuel && d+to_r[p]<=health;
            int ranks[]={d*256-g, (d+b)*256+b, (d+relaxed[0][p])*256+d,
                (d+nearest(p,sites[1],w))*256+d, (d+nearest(p,sites[2],w))*256+d,
                (back_f||back_r)?(d+b)*256+b:100000000, (9-g)*2048+d, b*512+d};
            for (int j=0;j<8;++j) if (ranks[j]<best[j]) {best[j]=ranks[j];picks[j]=p;}
        }
        int count=0;
        for (int j=0;j<8;++j) {
            int p=picks[j];
            if (p<0 || std::find(picks.begin(),picks.begin()+j,p)!=picks.begin()+j) continue;
            float* t=fronts+(a*8+count++)*12;
            t[0]=float(known[p]+1); t[1]=float(gain[p]); t[2]=float(cheb(p,goal,w));
            t[3]=float(relaxed[0][p]); t[4]=float(to_f[p]); t[5]=float(to_r[p]);
            t[6]=float(relaxed[1][p]); t[7]=float(relaxed[2][p]);
            t[8]=float(p%w-pos%w); t[9]=float(p/w-pos/w);
            t[10]=float(nearest(p,sites[1],w)); t[11]=float(nearest(p,sites[2],w));
        }
    }
}
