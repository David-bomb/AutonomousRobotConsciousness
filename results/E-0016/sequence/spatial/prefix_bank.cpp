// Частоты геодезических префиксов на независимых учебных дорогах.
// При применении передаются только типы клеток памяти и два текущих показания.
#include "../geometry.cpp"
#include <cstdint>
#include <fstream>
#include <random>
#include <unordered_map>
using Prefix=uint64_t;
constexpr int count_steps=12,max_meter=64;
struct Sym{int sx,sy;bool swap;int forward[8],back[8];};
const vector<Sym> maps=[](){
    vector<Sym> result;
    for(bool flip:{false,true})for(int sx:{-1,1})for(int sy:{-1,1}){
        Sym s{sx,sy,flip,{},{}};
        for(int a=0;a<8;a++)for(int b=0;b<8;b++)
            if(sx*(flip?dy[a]:dx[a])==dx[b]&&sy*(flip?dx[a]:dy[a])==dy[b]){
                s.forward[a]=b;s.back[b]=a;
            }
        result.push_back(s);
    }
    return result;
}();
int slot(int a,int b,int d,int parity,int group){return (((a*48+b)*65+d)*4+parity)*6+group;}
int band(int diff){return diff< -3?1:diff<0?2:diff<3?3:diff<7?4:5;}
int par(int x,int y,bool swap){return swap?((y&1)+2*(x&1)):((x&1)+2*(y&1));}
using Table=unordered_map<int,vector<pair<Prefix,uint32_t>>>;
using Counts=unordered_map<int,unordered_map<Prefix,uint32_t>>;

extern "C" __declspec(dllexport) void *create_counts(){return new Counts();}
extern "C" __declspec(dllexport) void train_prefix(void *address,const char *cells,int w,int h,uint32_t seed){
    Counts &bank=*reinterpret_cast<Counts*>(address);
    mt19937 rng(seed);
    Board physical(cells,cells,w,h,0);
    for(char kind:{'F','R'}){
        vector<int> sites;
        for(int i=0;i<w*h;i++)if(cells[i]==kind)sites.push_back(i);
        array<Dist,2> ds{physical.distances(sites[0]),physical.distances(sites[1])};
        for(int target=0;target<2;target++){
            int s=sites[target],other=sites[1-target];auto &length=ds[target];
            vector<int> next(w*h,-1);
            for(int i=0;i<w*h;i++)if(length[i]<=max_meter && length[i]>0){
                int n=0,picks[8];
                for(int a=0;a<8;a++){
                    int q=physical.edges[i][a];
                    if(q>=0 && length[q]+1==length[i])picks[n++]=a;
                }
                next[i]=picks[uniform_int_distribution<int>(0,n-1)(rng)];
            }
            for(int p=0;p<w*h;p++){
                int d=length[p];
                if(d<=0 || d>max_meter || d>ds[1-target][p])continue;
                int x=p%w,y=p/w,X=s%w-x,Y=s/w-y;
                int group=band(max(abs(other%w-x),abs(other/w-y))-max(abs(X),abs(Y)));
                int sequence[count_steps],u=p;
                for(int t=0;t<min(count_steps,d);t++){sequence[t]=next[u];u=physical.edges[u][next[u]];}
                for(auto &r:maps){
                    int A=r.sx*(r.swap?Y:X),B=r.sy*(r.swap?X:Y);
                    if(A<B||B<0)continue;
                    Prefix prefix=0;
                    for(int t=0;t<min(count_steps,d);t++)prefix|=Prefix(r.forward[sequence[t]])<<(3*t);
                    bank[slot(A,B,d,par(x,y,r.swap),0)][prefix]++;
                    bank[slot(A,B,d,par(x,y,r.swap),group)][prefix]++;
                }
            }
        }
    }
}
extern "C" __declspec(dllexport) int write_prefix(void *address,const char *path){
    auto *bank=reinterpret_cast<Counts*>(address);
    ofstream dst(path,ios::binary);int n=0;
    vector<int> keys;for(auto &r:*bank)keys.push_back(r.first);
    sort(keys.begin(),keys.end());
    for(int key:keys){
        vector<pair<Prefix,uint32_t>> bin((*bank)[key].begin(),(*bank)[key].end());
        sort(bin.begin(),bin.end());
        for(auto [s,c]:bin){
            uint32_t row[]={uint32_t(key),uint32_t(s),uint32_t(s>>32),c};
            dst.write((char*)row,sizeof row);n++;
        }
    }
    delete bank;
    return dst?n:-1;
}
extern "C" __declspec(dllexport) void *load_prefix(const uint32_t *rows,int count){
    auto *table=new Table();
    for(int i=0;i<count;i++,rows+=4)(*table)[rows[0]].push_back({rows[1]|(Prefix(rows[2])<<32),rows[3]});
    return table;
}
extern "C" __declspec(dllexport) void drop_prefix(void *addr){delete reinterpret_cast<Table*>(addr);}
extern "C" __declspec(dllexport) void score_prefix(const void *addr,const char *cells,int w,int h,int p,int d,int second,
                                                    char kind,float *out){
    fill(out,out+8*4,0.f);
    if(d<1||d>max_meter)return;
    const Table &table=*reinterpret_cast<const Table*>(addr);
    Board known(cells,cells,w,h,0),opened(cells,cells,w,h,1);
    auto root=known.distances(p);
    vector<int> sites,others;
    for(int i=0;i<w*h;i++){
        if(cells[i]==kind)sites.push_back(i);
        if(cells[i]==(kind=='F'?'R':'F'))others.push_back(i);
    }
    Dist upper(w*h,inf),foreign(w*h,inf);
    for(int s:sites){auto z=known.distances(s);for(int i=0;i<w*h;i++)upper[i]=min(upper[i],z[i]);}
    for(int s:others){auto z=known.distances(s);for(int i=0;i<w*h;i++)foreign[i]=min(foreign[i],z[i]);}
    int x=p%w,y=p/w;
    for(int k=0;k<2;k++){
        int s=sites[k],other=sites[1-k];
        auto lower=opened.distances(s);
        if(lower[p]>d)continue;
        int X=s%w-x,Y=s/w-y;
        int group=band(max(abs(other%w-x),abs(other/w-y))-max(abs(X),abs(Y)));
        for(const Sym &r:maps){
            int A=r.sx*(r.swap?Y:X),B=r.sy*(r.swap?X:Y);
            if(A<B||B<0)continue;
            unordered_map<Prefix,bool> valid;
            for(int a=max(0,A-1);a<=A+1;a++)for(int b=max(0,B-1);b<=min(a,B+1);b++){
                for(int dist=max(min(count_steps,d),d-2);dist<=min(max_meter,d+2);dist++){
                    int delta=abs(a-A)+abs(b-B)+abs(d-dist);
                    for(int g:{0,group}){
                        auto bin=table.find(slot(a,b,dist,par(x,y,r.swap),g));
                        if(bin==table.end())continue;
                        for(auto [code,count]:bin->second){
                            auto seen=valid.find(code);
                            bool ok=true;
                            if(seen!=valid.end())ok=seen->second;
                            else{
                                int u=p;
                                for(int t=1;t<=min(count_steps,d);t++){
                                    int step=r.back[(code>>(3*(t-1)))&7];
                                    u=opened.edges[u][step];
                                    if(u<0||root[u]<t||lower[u]>d-t||upper[u]<d-t||foreign[u]+t<second){
                                        ok=false;break;
                                    }
                                }
                                valid.emplace(code,ok);
                            }
                            if(ok){
                                float *v=out+4*r.back[code&7];
                                v[g?1:0]+=powf(.35f,(float)delta)*count;
                                if(!delta)v[g?3:2]+=count;
                            }
                        }
                    }
                }
            }
            break;
        }
    }
    for(int j=0;j<4;j++){
        float total=0;for(int a=0;a<8;a++)total+=out[4*a+j];
        if(total>0)for(int a=0;a<8;a++)out[4*a+j]/=total;
    }
}
