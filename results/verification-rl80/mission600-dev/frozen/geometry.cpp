// Все расстояния рассчитаны исключительно по переданной памяти типов клеток.
#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

using namespace std;
const int dx[] = {0,1,0,-1,1,1,-1,-1};
const int dy[] = {-1,0,1,0,-1,1,1,-1};
const int inf = 250;

// Три варианта трактовки неизвестных клеток: непроходимы, длина 1 или штраф +2.
extern "C" __declspec(dllexport) void lengths(const char *grid, int w, int h, int p,
                                                const int *targets, int count, float *out) {
    const int n = w*h, stride = 3*count+4;
    vector<int> edges(n*8, -1), hidden(n);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        int u=y*w+x;
        if (grid[u]=='#') continue;
        for (int m=0;m<8;m++) {
            int xx=x+dx[m], yy=y+dy[m];
            if (xx<0 || xx>=w || yy<0 || yy>=h) continue;
            int v=yy*w+xx;
            if (grid[v]=='#' || grid[y*w+xx]=='#' || grid[yy*w+x]=='#') continue;
            edges[8*u+m]=v;
            if (grid[v]=='?' || grid[y*w+xx]=='?' || grid[yy*w+x]=='?') hidden[u]|=(1<<m);
        }
    }
    for (int a=0;a<8;a++) {
        float *f=out+a*stride;
        fill(f, f+stride, (float)inf);
        int first=edges[8*p+a];
        if(first<0 || (hidden[p] & (1<<a))) continue;
        for(int mode=0; mode<3; mode++) {
            vector<int> d(n,inf);
            // Положительные целые веса; очереди по достигнутой длине.
            vector<vector<int>> buckets(inf);
            buckets[1].push_back(first);
            d[first]=1;
            int known=0, frontier=inf, furthest=0;
            int last=1;
            for(int l=1;l<=last;l++) {
                for(int u:buckets[l]) {
                    if(d[u]!=l) continue;
                    if(mode==0) {
                        known++;
                        furthest=l;
                        if(hidden[u]) frontier=min(frontier,l);
                    }
                    for(int m=0;m<8;m++) {
                        int v=edges[u*8+m];
                        if(v<0 || v==p) continue;
                        bool unobserved=(hidden[u]&(1<<m))!=0 || grid[u]=='?';
                        if(mode==0 && unobserved) continue;
                        int dl=l+1+(mode==2 && unobserved ? 2 : 0);
                        if(dl<d[v]) {
                            d[v]=dl;
                            buckets[dl].push_back(v);
                            last=max(last,dl);
                        }
                    }
                }
            }
            for(int t=0;t<count;t++) f[3*t+mode]=d[targets[t]];
            if(!mode) {
                f[3*count]=(float)frontier;
                f[3*count+1]=(float)known;
                f[3*count+2]=(float)furthest;
                f[3*count+3]=(float)(hidden[first]!=0);
            }
        }
    }
}
