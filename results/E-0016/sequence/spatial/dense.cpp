// Только учебная разметка. Клетки всех кратчайших путей из текущей позиции.
#include "region.cpp"
extern "C" __declspec(dllexport) int label_dense(const char *grid,int w,int h,int n,const int16_t *meta,uint8_t *label){
    Board graph(grid,grid,w,h,0);
    Dist ends[3]={Dist(w*h,inf),Dist(w*h,inf),Dist(w*h,inf)};
    for(int i=0;i<w*h;i++){
        int k=grid[i]=='F'?0:grid[i]=='R'?1:grid[i]=='B'?2:-1;
        if(k<0)continue;
        auto ds=graph.distances(i);
        for(int u=0;u<w*h;u++)ends[k][u]=min(ends[k][u],ds[u]);
    }
    fill(label,label+n*area,0);
    for(int j=0;j<n;j++){
        int pad=meta[8*j],p=(pad/side)*w+pad%side;
        if(ends[0][p]!=meta[8*j+6]||ends[1][p]!=meta[8*j+7])return -1;
        auto from=graph.distances(p);
        for(int i=0;i<w*h;i++)if(grid[i]!='#'){
            int bits=8;
            for(int k=0;k<3;k++)if(from[i]+ends[k][i]==ends[k][p])bits|=1<<k;
            label[j*area+i/w*side+i%w]=bits;
        }
    }
    return 0;
}
