// Свойства возможного первого выхода из известной дороги по текущей памяти.
#include "../geometry.cpp"
#include <cstdint>

namespace {
constexpr int side = 40, area = 1600, keep = 8, channels = 32;
struct Gate {
    int before, after, steps, target, lower, direction;
    double order;
};
int manhattan(int p, int q) { return abs(p%side-q%side)+abs(p/side-q/side); }
int chebyshev(int p, int q) { return max(abs(p%side-q%side),abs(p/side-q/side)); }
}

extern "C" __declspec(dllexport) void gate_features(const uint8_t *floor, const int16_t *meta,
         int urgent, int margin, int other_margin, float *out, int16_t *witness) {
    fill(out, out+8*keep*channels, 0.f);
    fill(witness, witness+8*keep*4, int16_t(-1));
    const int origin=meta[0], d=meta[6+urgent], other_d=meta[7-urgent];
    if(d<=0) return;
    vector<char> memory(area);
    for(int p=0;p<area;p++) memory[p]=floor[p]==0?'?':floor[p]==1?'.':'#';
    Board known(memory.data(),memory.data(),side,side,0);
    Board free(memory.data(),memory.data(),side,side,1);
    auto prefix=known.distances(origin);
    const int goal=meta[5], station[]={meta[1+2*urgent],meta[2+2*urgent]};
    const int other[]={meta[3-2*urgent],meta[4-2*urgent]};
    array<Dist,2> lower,upper;
    for(int t=0;t<2;t++){
        lower[t]=free.distances(station[t]);
        upper[t]=known.distances(station[t]);
    }
    auto other0=known.distances(other[0]), other1=known.distances(other[1]);
    array<vector<Gate>,8> candidates;
    for(int p=0;p<area;p++) {
        int len=prefix[p], mask=floor[area+p];
        if(!mask || len>d) continue;
        for(int t=0;t<2;t++){
            if(p==station[t] && len==d){
                for(int a=0;a<8;a++) if(mask&(1<<a))
                    candidates[a].push_back({p,p,len,t,0,-1,-1000.});
            }
            if(len>=d)continue;
            for(int k=0;k<8;k++){
                int q=free.edges[p][k];
                if(q<0 || known.edges[p][k]>=0 || prefix[q]<len+1) continue;
                int used=len+1, rem=d-used, bound=lower[t][q];
                if(bound>rem || min(upper[0][q],upper[1][q])<rem ||
                        used+min(other0[q],other1[q])<other_d) continue;
                int man=manhattan(q,station[t]);
                double order=abs(rem-man*1.35)+.09*bound-.05*used;
                for(int a=0;a<8;a++) if(mask&(1<<a))
                    candidates[a].push_back({p,q,used,t,bound,k,order});
            }
        }
    }
    for(int a=0;a<8;a++){
        auto &list=candidates[a];
        stable_sort(list.begin(),list.end(),[](const Gate &u,const Gate &v){return u.order<v.order;});
        int count=0;
        vector<int> endpoints;
        for(auto &g:list){
            int id=2*g.after+g.target;
            if(find(endpoints.begin(),endpoints.end(),id)!=endpoints.end()) continue;
            endpoints.push_back(id);
            float *x=out+(a*keep+count)*channels;
            int16_t *label=witness+(a*keep+count)*4;
            label[0]=g.before;label[1]=g.after;label[2]=g.steps;label[3]=station[g.target];
            int s=station[g.target],q=g.after,rem=d-g.steps;
            int rx=s%side-q%side,ry=s/side-q/side;
            int ex=q%side-origin%side,ey=q/side-origin/side;
            bool fixed=g.before==q;
            x[0]=1;x[1]=fixed;x[2]=d/35.f;x[3]=rem/35.f;x[4]=g.steps/30.f;
            x[5]=g.lower/35.f;x[6]=(rem-g.lower)/25.f;
            x[7]=(rem-manhattan(q,s))/20.f;x[8]=abs(rx)/30.f;x[9]=abs(ry)/30.f;
            x[10]=min(abs(rx),abs(ry))/20.f;x[11]=manhattan(origin,s)/40.f;
            x[12]=manhattan(q,other[0])<manhattan(q,other[1])?manhattan(q,other[0])/40.f:manhattan(q,other[1])/40.f;
            x[13]=other_d/45.f;x[14]=clamp(margin,-20,30)/15.f;x[15]=clamp(other_margin,-20,50)/25.f;
            x[16]=chebyshev(q,goal)/35.f;x[17]=manhattan(q,station[1-g.target])/40.f;
            x[18]=(dx[a]*ex+dy[a]*ey)/20.f;x[19]=(dx[a]*rx+dy[a]*ry)/25.f;
            x[20]=fixed?0:(dx[g.direction]*rx+dy[g.direction]*ry)/25.f;
            x[21]=fixed?0:(dx[g.direction]*ex+dy[g.direction]*ey)/25.f;
            x[22]=abs(q%side)%2;x[23]=(q/side)%2;x[24]=a>=4;x[25]=g.direction>=4;
            x[26]=min(int(list.size()),60)/40.f;
            x[27]=min(min(upper[0][q],upper[1][q]),80)/60.f;
            int unknown=0,blocked=0,open=0;
            for(int iy=-2;iy<=2;iy++)for(int ix=-2;ix<=2;ix++){
                int xx=q%side+ix, yy=q/side+iy;
                if(xx<0||xx>=side||yy<0||yy>=side){blocked++;continue;}
                int f=floor[yy*side+xx];unknown+=f==0;open+=f==1;blocked+=f==2;
            }
            x[28]=unknown/25.f;x[29]=blocked/25.f;x[30]=open/25.f;
            x[31]=clamp(chebyshev(origin,s)-chebyshev(q,s),-30,30)/25.f;
            if(++count==keep)break;
        }
    }
}
