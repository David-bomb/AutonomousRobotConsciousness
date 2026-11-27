// Учебные метки на независимых дорогах; этот модуль отсутствует в адаптере модели.
#include "region.cpp"
#include <random>

extern "C" __declspec(dllexport) void generate_samples(const char *grid,int w,int h,int count,
               uint32_t seed,uint8_t *images,int16_t *metadata,int8_t *truth) {
    Board board(grid,grid,w,h,0);
    int start=0,goal=0;
    vector<int> pos,stations[2],eligible;
    for(int i=0;i<w*h;i++){
        if(grid[i]=='F')stations[0].push_back(i);
        if(grid[i]=='R')stations[1].push_back(i);
        if(grid[i]=='A')start=i;
        if(grid[i]=='B')goal=i;
        if(grid[i]!='#'&&grid[i]!='B'){
            pos.push_back(i);
            if(count_if(board.edges[i].begin(),board.edges[i].end(),[](int q){return q>=0;})>1)eligible.push_back(i);
        }
    }
    Dist d[3]={Dist(w*h,inf),Dist(w*h,inf),board.distances(goal)};
    for(int k=0;k<2;k++)for(int s:stations[k]){
        auto ds=board.distances(s);for(int i=0;i<w*h;i++)d[k][i]=min(d[k][i],ds[i]);
    }
    auto from_start=board.distances(start);
    vector<int> near;
    for(int u:eligible)if(from_start[u]<=20)near.push_back(u);
    mt19937 rng(seed);
    auto draw=[&](int n){return uniform_int_distribution<int>(0,n-1)(rng);};
    for(int j=0;j<count;j++){
        int src=j%6==0?start:(j%3==0?near[draw((int)near.size())]:eligible[draw((int)eligible.size())]);
        vector<char> mem(w*h,'?');
        for(int k=0;k<2;k++)for(int s:stations[k])mem[s]=grid[s];
        auto reveal=[&](int u){
            for(int i=max(0,u%w-1);i<=min(w-1,u%w+1);i++)
                for(int y=max(0,u/w-1);y<=min(h-1,u/w+1);y++)mem[y*w+i]=grid[y*w+i];
        };
        int budgets[]={0,1,3,5,8,15,25,40,70,120,180,260};
        int budget=budgets[draw(12)];
        vector<uint8_t> visited(w*h,0);
        vector<int> open{src};
        int p=src;
        reveal(src);visited[src]=1;
        for(int step=0;step<budget&&!open.empty();step++){
            int index=j%2?int(open.size())-1:draw((int)open.size());
            p=open[index];
            vector<int> candidates;
            for(int v:board.edges[p])if(v>=0&&v!=goal&&!visited[v])candidates.push_back(v);
            if(candidates.empty()){open[index]=open.back();open.pop_back();continue;}
            p=candidates[draw((int)candidates.size())];
            reveal(p);visited[p]=1;open.push_back(p);
        }
        // Точка запроса могла быть последней точкой обратной физической прогулки,
        // которая открыла именно такое связное множество окон.
        region(mem.data(),w,h,src,images+j*2*area);
        auto pad=[&](int u){return (u/w)*side+u%w;};
        auto m=metadata+j*8;
        m[0]=pad(src);m[1]=pad(stations[0][0]);m[2]=pad(stations[0][1]);
        m[3]=pad(stations[1][0]);m[4]=pad(stations[1][1]);m[5]=pad(goal);
        m[6]=d[0][src];m[7]=d[1][src];
        for(int a=0;a<8;a++){
            int q=board.edges[src][a];
            for(int k=0;k<3;k++)truth[(j*8+a)*3+k]=q<0?-1:1+d[k][q]-d[k][src];
        }
    }
}
