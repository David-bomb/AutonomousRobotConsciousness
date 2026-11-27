// Длины только в видимой и в оптимистически открытой памяти между разрешёнными ориентирами.
#include "../geometry.cpp"
#include <cstdint>

extern "C" __declspec(dllexport) void landmark_features(const uint8_t *board,const int16_t *meta,
                                                         float *pairs,float *actions){
    constexpr int side=40,n=1600;
    vector<char> cells(n);
    for(int i=0;i<n;i++)cells[i]=board[i]==0?'?':board[i]==1?'.':'#';
    Board known(cells.data(),cells.data(),side,side,0),possible(cells.data(),cells.data(),side,side,1);
    array<Dist,6> low,high;
    array<float,6> reveal={},degree={};
    array<int,8> owned={};
    int opened=0;
    for(int p=0;p<n;p++) if(board[n+p]){
        opened++;
        for(int a=0;a<8;a++)owned[a]+=bool(board[n+p]&(1<<a));
    }
    for(int i=0;i<6;i++){
        int p=meta[i],x=p%side,y=p/side;
        low[i]=possible.distances(p);high[i]=known.distances(p);
        for(int dy0=-2;dy0<=2;dy0++)for(int dx0=-2;dx0<=2;dx0++){
            int X=x+dx0,Y=y+dy0;
            reveal[i]+=(X<0||Y<0||X>=side||Y>=side||board[Y*side+X]!=0)/25.f;
        }
        for(int q:known.edges[p])degree[i]+=(q>=0)/8.f;
    }
    auto type=[](int i){return i==0?0:i<3?1:i<5?2:3;};
    for(int i=0;i<6;i++) for(int j=0;j<6;j++){
        int p=meta[i],t=meta[j];
        int x=abs(p%side-t%side),y=abs(p/side-t/side);
        float *f=pairs+(6*i+j)*18;
        int l=low[i][t],u=high[i][t];
        if(i==0 && (type(j)==1||type(j)==2))l=max(l,int(meta[5+type(j)]));
        if(j==0 && (type(i)==1||type(i)==2))l=max(l,int(meta[5+type(i)]));
        f[0]=min(l,127)/127.f;f[1]=min(u,127)/127.f;f[2]=u<inf;f[3]=max(x,y)/36.f;
        f[4]=min(x,y)/30.f;f[5]=reveal[i];f[6]=reveal[j];f[7]=degree[i];f[8]=degree[j];
        f[9]=type(i)/3.f;f[10]=type(j)/3.f;f[11]=((p%side)%2+(p/side)%2)/2.f;
        f[12]=((t%side)%2+(t/side)%2)/2.f;f[13]=meta[6]/45.f;f[14]=meta[7]/45.f;
        f[15]=min(opened,500)/500.f;f[16]=min(low[0][p],127)/127.f;f[17]=min(low[0][t],127)/127.f;
    }
    fill(actions,actions+8*5*16,0.f);
    int origin=meta[0];
    for(int a=0;a<8;a++){
        int q=known.edges[origin][a];
        if(q<0)continue;
        auto opt=possible.distances(q,origin,1),actual=known.distances(q,origin,1);
        for(int t=1;t<6;t++){
            int site=meta[t],rx=site%side-origin%side,ry=site/side-origin/side;
            float *f=actions+(a*5+t-1)*16;
            f[0]=1;f[1]=min(opt[site],127)/127.f;f[2]=min(actual[site],127)/127.f;
            f[3]=actual[site]<inf;f[4]=clamp(opt[site]-low[0][site],0,20)/10.f;
            f[5]=actual[site]<inf?(actual[site]-high[0][site])/20.f:0;
            f[6]=(dx[a]*rx+dy[a]*ry)/50.f;f[7]=abs(dx[a]*ry-dy[a]*rx)/50.f;
            f[8]=a>=4;f[9]=float(owned[a])/max(1,opened);
            f[10]=(low[t][q]-low[t][origin]);f[11]=high[t][q]<inf?(high[t][q]-high[t][origin]):0;
            f[12]=int(bool(board[n+site]&(1<<a)));f[13]=degree[t];f[14]=reveal[t];
            int num=0;for(int next:known.edges[q])num+=next>=0;f[15]=num/8.f;
        }
    }
}
