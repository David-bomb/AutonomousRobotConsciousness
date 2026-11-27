// Совместные признаки достижимой границы, запасов после прибытия и последующего
// маршрута до B. Источник сведений о дорогах — один снимок открытых клеток.
#include "../geometry.cpp"
struct Tail {int length,need_f,need_h;};

extern "C" __declspec(dllexport) void itineraries(const char *grid,const char *guessed,int w,int h,int p,
                              const int *sites,int f,int hstock,float *out){
    const int n=w*h;
    fill(out,out+8*3*3*4,(float)inf);
    Board known(grid,guessed,w,h,0);
    array<Dist,5> located;
    for(int s=0;s<5;s++) located[s]=known.distances(sites[s]);
    vector<int> boundary;
    for(int b=0;b<n;b++){
        if(grid[b]=='#'||grid[b]=='?'||b==p)continue;
        int x=b%w,y=b/w;
        for(int k=0;k<8;k++){
            int X=x+dx[k],Y=y+dy[k];
            if(X>=0 && X<w && Y>=0 && Y<h && grid[Y*w+X]=='?'){boundary.push_back(b);break;}
        }
    }
    int kind[]={0,1,1,2,2};
    for(int mode=1;mode<=3;mode++){
        Board future(grid,guessed,w,h,mode);
        array<Dist,5> predicted;
        for(int s=0;s<5;s++)predicted[s]=future.distances(sites[s]);
        vector<vector<Tail>> tails(boundary.size());
        for(int j=0;j<(int)boundary.size();j++){
            int b=boundary[j];
            auto route=[&](initializer_list<int> stops){
                Stocks full{100,100};
                int used=0,rf=-1,rh=-1,last=b;
                for(int s:stops){
                    int d=predicted[s][last];
                    if(d>=inf || (!d && s!=0) || arrive(full,kind[s],d)<0)return;
                    used+=d;
                    if((kind[s]==1||s==0)&&rf<0)rf=used;
                    if((kind[s]==2||s==0)&&rh<0)rh=used;
                    last=sites[s];
                }
                // Удаляем только доминируемые варианты: та же/меньшая длина
                // и не большие требования к обоим начальным запасам.
                for(auto t:tails[j])if(t.length<=used&&t.need_f<=rf&&t.need_h<=rh)return;
                tails[j].erase(remove_if(tails[j].begin(),tails[j].end(),[&](Tail t){
                    return used<=t.length&&rf<=t.need_f&&rh<=t.need_h;}),tails[j].end());
                tails[j].push_back({used,rf,rh});
            };
            route({0});
            for(int i=1;i<=4;i++){
                route({i,0});
                for(int k=1;k<=4;k++)if(kind[k]!=kind[i])route({i,k,0});
            }
        }
        for(int a=0;a<8;a++){
            int q=known.edges[p][a];
            if(q<0)continue;
            Dist from=known.distances(q,p,1);
            for(int j=0;j<(int)boundary.size();j++){
                int b=boundary[j];
                auto assess=[&](Stocks s,int moved,int d){
                    if(d>=inf||moved+d>=inf)return;
                    int local=0;
                    for(int i=1;i<=4;i++)if(sites[i]==b)local=kind[i];
                    if(d>0 && arrive(s,local,d)<(local?0:1))return;
                    int safety=0;
                    for(int i=1;i<=4;i++){
                        int back=located[i][b];
                        Stocks at=s;
                        if(back>=inf || (back>0 && arrive(at,kind[i],back)<0))continue;
                        safety=max(safety,1);
                        for(int k=1;k<=4;k++)if(kind[k]!=kind[i]){
                            auto at2=at;
                            if(located[i][sites[k]]<inf&&arrive(at2,kind[k],located[i][sites[k]])>=0)safety=2;
                        }
                    }
                    int reached=moved+d;
                    for(auto t:tails[j]){
                        int margin=min(s.f-t.need_f,s.h-t.need_h);
                        if(margin<0)continue;
                        for(int c=0;c<=safety;c++){
                            float *v=out+((a*3+mode-1)*3+c)*4;
                            v[0]=min(v[0],(float)(reached+t.length));
                            v[1]=min(v[1],(float)(reached*2+t.length));
                            v[2]=min(v[2],(float)reached);
                            v[3]=min(v[3],(float)(-margin));
                        }
                    }
                };
                assess({f,hstock},0,from[b]);
                for(int i=1;i<=4;i++){
                    int d=from[sites[i]];
                    Stocks s{f,hstock};
                    if(d>=inf||arrive(s,kind[i],d)<0)continue;
                    assess(s,d,located[i][b]);
                    for(int k=1;k<=4;k++)if(kind[k]!=kind[i]){
                        int hop=located[i][sites[k]];
                        Stocks after=s;
                        if(hop>=inf||arrive(after,kind[k],hop)<0)continue;
                        assess(after,d+hop,located[k][b]);
                    }
                }
            }
        }
    }
}
