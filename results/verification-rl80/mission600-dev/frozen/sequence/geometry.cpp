// Длины для всей последовательности и возврата от границы. Только наблюдаемая карта.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
using namespace std;
const int inf=300, dx[]={0,1,0,-1,1,1,-1,-1}, dy[]={-1,0,1,0,-1,1,1,-1};
const int modes=4, orders=5, order_fields=4, frontier_fields=12;
const int stride=modes*orders*order_fields+frontier_fields;
using Dist=vector<int>;

struct Board {
    int w,h;
    vector<array<int,8>> edges,weight;
    Board(const char *grid,const char *guessed,int width,int height,int mode):w(width),h(height),
            edges(w*h),weight(w*h) {
        auto wall=[&](int p){return grid[p]=='#' || (mode==0 && grid[p]=='?') ||
                                    (mode>=2 && guessed[p]=='#');};
        for(int y=0;y<h;y++) for(int x=0;x<w;x++){
            int p=y*w+x;
            edges[p].fill(-1);
            if(wall(p)) continue;
            for(int k=0;k<8;k++){
                int X=x+dx[k],Y=y+dy[k];
                if(X<0 || X>=w || Y<0 || Y>=h) continue;
                int q=Y*w+X,u=y*w+X,v=Y*w+x;
                if(wall(q)||wall(u)||wall(v)) continue;
                edges[p][k]=q;
                weight[p][k]=1+(mode==3 && (grid[q]=='?' || grid[p]=='?' || grid[u]=='?' || grid[v]=='?'));
            }
        }
    }
    Dist distances(int source,int skip=-1,int initial=0) const {
        Dist d(w*h,inf);
        vector<vector<int>> queue(inf);
        d[source]=initial;queue[initial].push_back(source);
        int far=initial;
        for(int len=initial;len<=far;len++){
            for(int p:queue[len]){
                if(d[p]!=len) continue;
                for(int k=0;k<8;k++){
                    int q=edges[p][k];
                    if(q<0 || q==skip) continue;
                    int next=len+weight[p][k];
                    if(next<d[q]){
                        d[q]=next;queue[next].push_back(q);far=max(far,next);
                    }
                }
            }
        }
        return d;
    }
};

struct Stocks{int f,h;};
int arrive(Stocks &s,int kind,int d){
    s.f-=d;s.h-=d;
    int margin=min(s.f,s.h);
    if(kind==1){margin=min(s.f,s.h-1);s.f=100;}
    if(kind==2){margin=min(s.h,s.f-1);s.h=100;}
    return margin;
}

extern "C" __declspec(dllexport) void sequences(const char *grid,const char *guessed,int w,int h,int p,
                          const int *targets,int f,int hstock,int df,int dr,float *output){
    // targets[0] — цель, 1/2 — сервис меньшего запаса, 3/4 — другого.
    int kind[]={0,1,1,2,2};
    vector<int> boundary;
    for(int i=0;i<w*h;i++){
        if(grid[i]=='#' || grid[i]=='?' || i==p) continue;
        int x=i%w,y=i/w;
        bool fresh=false;
        for(int j=0;j<8;j++){
            int X=x+dx[j],Y=y+dy[j];
            if(X>=0 && X<w && Y>=0 && Y<h && grid[Y*w+X]=='?') fresh=true;
        }
        if(fresh) boundary.push_back(i);
    }
    fill(output,output+8*stride,0.f);
    for(int mode=0;mode<modes;mode++){
        Board board(grid,guessed,w,h,mode);
        array<Dist,5> site;
        for(int i=0;i<5;i++) site[i]=board.distances(targets[i]);
        for(int a=0;a<8;a++){
            float *out=output+a*stride;
            int q=board.edges[p][a];
            if(q<0) continue;
            auto from=board.distances(q,p,1);
            int first[5];
            for(int i=0;i<5;i++) first[i]=max(from[targets[i]],i==0?0:(kind[i]==1?df:dr));
            for(int order=0;order<orders;order++){
                int min_length=inf,max_margin=-inf,min_feasible=inf,valid=0;
                auto route=[&](vector<int> stops){
                    Stocks stocks{f,hstock};int last=-1,length=0,margin=inf;
                    for(int i:stops){
                        int d=last<0?first[i]:site[last][targets[i]];
                        if(d>=inf || (d==0 && last>=0)) return;
                        length+=d;last=i; margin=min(margin,arrive(stocks,kind[i],d));
                    }
                    min_length=min(min_length,length);max_margin=max(max_margin,margin);
                    if(margin>=0){min_feasible=min(min_feasible,length);valid++;}
                };
                if(order==0) route({0});
                for(int i=1;i<=4;i++){
                    if(order==kind[i]) route({i,0});
                    if(order==kind[i]+2)
                        for(int j=1;j<=4;j++) if(kind[i]!=kind[j]) route({i,j,0});
                }
                int start=(mode*orders+order)*order_fields;
                out[start]=float(min_length);out[start+1]=float(max_margin);
                out[start+2]=float(min_feasible);out[start+3]=float(valid);
            }
            if(mode) continue;
            float *survey=out+modes*orders*order_fields;
            fill(survey,survey+frontier_fields,(float)inf);
            for(int b:boundary){
                int estimate=max(abs(b%w-targets[0]%w),abs(b/w-targets[0]/w));
                auto visit=[&](int used,Stocks s){
                    if(used>=inf || min(s.f,s.h)<=0) return;
                    bool back_one=false,back_both=false;
                    for(int i=1;i<=4;i++){
                        int d=site[i][b];if(d>=inf) continue;
                        Stocks ret=s;if(arrive(ret,kind[i],d)<0) continue;
                        back_one=true;
                        for(int j=1;j<=4;j++) if(kind[j]!=kind[i] && site[i][targets[j]]<inf){
                            Stocks ret2=ret;
                            if(arrive(ret2,kind[j],site[i][targets[j]])>=0) back_both=true;
                        }
                    }
                    for(int t=0;t<3;t++) if(t==0 || (t==1?back_one:back_both)){
                        auto v=survey+4*t;
                        v[0]=min(v[0],(float)used);v[1]=min(v[1],(float)(used+estimate));
                        v[2]=min(v[2],(float)estimate);v[3]=min(v[3],(float)(used+2*estimate));
                    }
                };
                int d=from[b];
                if(d<inf) visit(d,{f-d,hstock-d});
                // Дойти до открываемого окна можно и после одного или двух заездов.
                for(int i=1;i<=4;i++){
                    int leg=first[i];Stocks s{f,hstock};
                    if(leg>=inf || arrive(s,kind[i],leg)<0) continue;
                    int to=site[i][b];
                    if(to<inf) visit(leg+to,{s.f-to,s.h-to});
                    for(int j=1;j<=4;j++) if(kind[i]!=kind[j]){
                        int hop=site[i][targets[j]];Stocks after=s;
                        if(hop>=inf || arrive(after,kind[j],hop)<0) continue;
                        to=site[j][b];
                        if(to<inf) visit(leg+hop+to,{after.f-to,after.h-to});
                    }
                }
            }
        }
    }
}
