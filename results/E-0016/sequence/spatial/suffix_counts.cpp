// Оценивание на учебных геометриях распределения длины продолжения за известной границей.
#include "../geometry.cpp"
#include <cstdint>

namespace {
// Последний столбец содержит число допустимых для разметки отправных клеток.
constexpr int forward_n=145, lateral_n=73, lengths_n=66, total=65;
int bin(int px,int py,int x,int y,int tx,int ty){
    int ux=x-px, uy=y-py, rx=tx-x, ry=ty-y;
    int forward=ux*rx+uy*ry, lateral=abs(ux*ry-uy*rx);
    bool diag=bool(ux && uy);
    int along=ux ? px%2 : py%2, across=ux ? py%2 : px%2;
    if(diag && along>across)swap(along,across);
    return ((((diag*4+2*along+across)*forward_n+forward+72)*lateral_n+lateral)*lengths_n);
}
}

extern "C" __declspec(dllexport) void learn_suffix(const char *grid,int w,int h,int32_t *counts) {
    Board truth(grid,grid,w,h,0);
    for(int station=0;station<w*h;station++){
        if(grid[station]!='F' && grid[station]!='R')continue;
        auto dist=truth.distances(station);
        int tx=station%w,ty=station/w;
        for(int p=0;p<w*h;p++)if(grid[p]!='#'){
            int px=p%w,py=p/w;
            for(int a=0;a<8;a++){
                int x=px+dx[a],y=py+dy[a];
                if(x<=0||y<=0||x>=w-1||y>=h-1)continue;
                int slot=bin(px,py,x,y,tx,ty);
                counts[slot+total]++;
                int q=truth.edges[p][a];
                if(q>=0 && dist[p]==dist[q]+1){
                    counts[slot+min(64,dist[q])]++;
                }
            }
        }
    }
}
