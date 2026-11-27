// Доступные направления к границам изученной дороги и возможные продолжения.
// Оба массива клеток получены исключительно из текущей памяти.
#include "geometry.cpp"

extern "C" __declspec(dllexport) void alternatives(const char *grid,const char *guess,
        int w,int h,int pos,const int *sites,int f,int r,int df,int dr,float *buf){
    const int n=w*h, fields=22;
    fill(buf,buf+8*5*fields,0.f);
    Board known(grid,guess,w,h,0), hopeful(grid,guess,w,h,2);
    Dist root=known.distances(pos);
    // Возможные продолжения, не использующие вновь изученную связную часть.
    // На графе с циклами это лишь одна гипотеза. Сеть также видит обычные длины.
    vector<char> outside(guess,guess+n);
    for(int i=0;i<n;i++) if(root[i]<inf) outside[i]='#';
    Board hidden(outside.data(),outside.data(),w,h,1);
    for(int t=0;t<5;t++){
        int goal=sites[t],meter=t==0?0:(t<3?df:dr),stock=t==0?min(f,r):(t<3?f:r);
        auto optimistic=hopeful.distances(goal),unseen=hidden.distances(goal);
        for(int a=0;a<8;a++){
            int q=known.edges[pos][a];
            if(q<0) continue;
            auto reach=known.distances(q,pos,1);
            float *out=buf+(a*5+t)*fields;
            int known_target=reach[goal],best=inf,best_raw=inf,closest=inf,cheapest=inf;
            int total=0,within=0,parity=0,budget=0,longest=0,nrevealed=0;
            float invsum=0,weighted=0;
            int largest_reserve=-inf, nearest_front=inf, min_goal=inf;
            for(int b=0;b<n;b++){
                if(reach[b]==inf) continue;
                int unknown_count=0,unused=inf,relaxed=inf;
                for(int k=0;k<8;k++){
                    int u=hopeful.edges[b][k];
                    if(u<0 || grid[u]!='?') continue;
                    unknown_count++;
                    unused=min(unused,unseen[u]+1);
                    relaxed=min(relaxed,optimistic[u]+1);
                }
                if(!unknown_count) continue;
                nrevealed+=unknown_count;
                nearest_front=min(nearest_front,reach[b]);
                min_goal=min(min_goal,relaxed);
                cheapest=min(cheapest,reach[b]+relaxed);
                best_raw=min(best_raw,relaxed);
                if(unused>=inf) continue;
                int length=reach[b]+unused;
                total++;longest=max(longest,reach[b]);
                best=min(best,length); closest=min(closest,unused);
                largest_reserve=max(largest_reserve,stock-length);
                if(length<=stock) budget++;
                if(length<=meter){
                    within++;parity+=((meter-length)%2==0);
                    float prob=1.f/(1+abs(length-meter)*.25f);
                    invsum+=prob;weighted+=prob*reach[b];
                }
            }
            auto len=[](int d){return min(d,120)/60.f;};
            auto residual=[&](int d,int ref){return clamp(d-ref,-30,30)/15.f;};
            out[0]=len(known_target);out[1]=known_target==meter;out[2]=root[goal]<inf;
            out[3]=len(best);out[4]=residual(best,meter);out[5]=residual(best,stock);
            out[6]=len(cheapest);out[7]=residual(cheapest,meter);out[8]=len(closest);
            out[9]=len(nearest_front);out[10]=len(min_goal);out[11]=log1pf((float)total)/4;
            out[12]=log1pf((float)within)/4;out[13]=log1pf((float)parity)/4;out[14]=log1pf((float)budget)/4;
            out[15]=log1pf((float)nrevealed)/4;out[16]=len(longest);out[17]=invsum?weighted/invsum/30:0;
            out[18]=log1pf(invsum)/4;out[19]=clamp(largest_reserve,-30,30)/15.f;
            out[20]=meter/60.f;out[21]=stock/100.f;
        }
    }
}
