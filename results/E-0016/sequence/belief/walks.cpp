// Суммы весов путей заданной длины. В граф входят только известные клетки и
// вероятности дороги, полученные от учебной выборки. Датчики заданы в начале пути.
#include "../geometry.cpp"

extern "C" __declspec(dllexport) void walks(const char *grid,int w,int h,int p,
        int d,int other,char target,const float *prob,float *out){
    const int n=w*h;
    fill(out,out+8*6,0.f);
    if(d<1 || d>=100) return;
    Board known(grid,grid,w,h,0), possible(grid,grid,w,h,1);
    auto from=known.distances(p);
    vector<int> true_goal(n,0), sites,other_sites;
    for(int i=0;i<n;i++){
        if(grid[i]==target){true_goal[i]=1;sites.push_back(i);}
        if(grid[i]==(target=='F'?'R':'F')) other_sites.push_back(i);
    }
    vector<int> upper(n,inf),lower(n,inf),second(n,inf);
    for(int s:sites){
        Dist u=known.distances(s),l=possible.distances(s);
        for(int i=0;i<n;i++){upper[i]=min(upper[i],u[i]);lower[i]=min(lower[i],l[i]);}
    }
    for(int s:other_sites){
        Dist v=known.distances(s);
        for(int i=0;i<n;i++) second[i]=min(second[i],v[i]);
    }
    // Три сглаживания вероятности неизвестного ребра, без выбора направления.
    for(int mode=0;mode<3;mode++){
        vector<array<float,8>> edge(n),current(n),next(n);
        for(int i=0;i<n;i++){
            for(int a=0;a<8;a++){
                int q=possible.edges[i][a];
                if(q<0)continue;
                float chance=prob[q];
                if(a>=4)chance*=prob[(i/w)*w+q%w]*prob[(q/w)*w+i%w];
                if(chance<.002f)continue;
                edge[i][a]=mode==0?chance:(mode==1?sqrtf(chance):chance*chance);
            }
        }
        const int inv[]={2,3,0,1,6,7,4,5};
        // current[u,a] — масса продолжений из u, если в u пришли направлением a.
        for(int s:sites) current[s].fill(1.f);
        for(int remaining=1;remaining<d;remaining++){
            int t=d-remaining;
            for(int i=0;i<n;i++){
                next[i].fill(0.f);
                if(true_goal[i]||from[i]<t||upper[i]<remaining||lower[i]>remaining||second[i]+t<other)continue;
                float terms[8]={},sum=0;
                for(int a=0;a<8;a++) if(edge[i][a]>0){
                    int q=possible.edges[i][a];
                    terms[a]=edge[i][a]*current[q][a];sum+=terms[a];
                }
                for(int prev=0;prev<8;prev++) next[i][prev]=max(0.f,sum-terms[inv[prev]]);
            }
            float mx=0;
            for(auto &row:next)for(float v:row)mx=max(mx,v);
            if(mx>1e7f)for(auto &row:next)for(float &v:row)v/=mx;
            swap(current,next);
        }
        float sum=0;
        for(int a=0;a<8;a++){
            int q=known.edges[p][a];
            if(q<0)continue;
            float value=edge[p][a]*current[q][a];
            out[a*6+mode]=value;sum+=value;
        }
        for(int a=0;a<8;a++){
            float &v=out[a*6+mode];
            out[a*6+3+mode]=v>0;
            if(sum>0)v/=sum;
        }
    }
}
