// Кодирование видимых дорог и их связи с каждым первым движением.
#include "../geometry.cpp"
#include <cstdint>
const int side=40,area=side*side;

extern "C" __declspec(dllexport) void region(const char *memory,int w,int h,int start,uint8_t *out) {
    Board board(memory,memory,w,h,0);
    fill(out,out+area,2);
    fill(out+area,out+2*area,0);
    vector<int> dist(w*h,-1),layer;
    vector<uint8_t> first(w*h,0);
    auto ix=[&](int p){return (p/w)*side+p%w;};
    for(int i=0;i<w*h;i++)out[ix(i)]=memory[i]=='?'?0:memory[i]=='#'?2:1;
    dist[start]=0;
    for(int a=0;a<8;a++) {
        int q=board.edges[start][a];
        if(q<0)continue;
        dist[q]=1;first[q]=1<<a;layer.push_back(q);
    }
    for(int k=1;!layer.empty();k++){
        vector<int> next;
        for(int u:layer){
            out[area+ix(u)]=first[u];
            for(int v:board.edges[u]){
                if(v<0)continue;
                if(dist[v]<0){dist[v]=k+1;next.push_back(v);}
                if(dist[v]==k+1)first[v]|=first[u];
            }
        }
        swap(layer,next);
    }
}
