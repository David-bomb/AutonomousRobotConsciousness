// Infer possible service missions of a sampled terrain before and after a first refill.
#include "morphology_worlds.cpp"
namespace {
bool site_visible(const char* m,int w,int p){
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++)if(m[p+w*y+x]=='?')return false;
    return true;
}
struct History {
    const int* starts;int count,pos,w,h,fuel,hp;
    bool first,cardinal=true;
    array<bool,5> visibility{};
    History(const char* m,int w,int h,int pos,const int* sites,const int* starts,int count,int f,int hp,bool first):
        starts(starts),count(count),pos(pos),w(w),h(h),fuel(f),hp(hp),first(first){
        for(int s=1;s<5;s++)visibility[s]=site_visible(m,w,sites[s]);
        auto known=[](char c){return c!='?'&&c!='#';};
        for(int y=0;y<h-1&&cardinal;y++)for(int x=0;x<w-1;x++){
            int p=y*w+x;
            if(known(m[p])&&known(m[p+1])&&known(m[p+w])&&known(m[p+w+1])){cardinal=false;break;}
        }
    }
    bool time_ok(int a,int elapsed,int bound)const{
        return elapsed>=bound && (!first||elapsed==0) && (!cardinal ||
            ((elapsed-abs(a%w-pos%w)-abs(a/w-pos/w))%2==0));
    }
    double support(const int* sites,const Fields& d)const{
        double best=0.;
        for(auto tier:tiers){
            if(outside(w/2,tier.low,tier.high)||outside(h/2,tier.low,tier.high))continue;
            for(int idx=0;idx<count;idx++){
                int a=starts[3*idx],bound=starts[3*idx+1],shortest=starts[3*idx+2],direct=d[0][a];
                if(outside(direct,tier.direct,tier.direct+26))continue;
                for(int i=1;i<5;i++){
                    int firstlen=d[i][a],lower_kind=(i<3?0:1);
                    if(outside(firstlen,4,min(27,direct/2)) || firstlen+d[0][sites[i]]<direct+2)continue;
                    for(int j=lower_kind?1:3;j<(lower_kind?3:5);j++){
                        int middle=d[i][sites[j]],end=d[0][sites[j]];
                        if(outside(middle,4,43)||middle+end>99||end>93)continue;
                        bool decoys=true;
                        for(int s=1;s<5;s++)if(s!=i&&s!=j)decoys&=(d[s][a]>=4 && d[s][a]+d[0][sites[s]]>direct);
                        if(!decoys)continue;
                        for(int m1=tier.m1lo;m1<=tier.m1hi;m1++)for(int m2=tier.m2lo;m2<=tier.m2hi;m2++){
                            int small=firstlen+m1,big=firstlen+middle+m2;
                            if(small>=direct || big>=min(direct,100))continue;
                            int f0=lower_kind?big:small,h0=lower_kind?small:big;
                            auto accept=[&](int elapsed){
                                if(time_ok(a,elapsed,bound))best=max(best,exp(-.16*(elapsed-bound)-.11*(bound-shortest)));
                            };
                            if(f0-fuel==h0-hp)accept(f0-fuel);
                            for(int s=1;s<5;s++)if(visibility[s]){
                                int total=s<3?h0-hp:f0-fuel,since=s<3?100-fuel:100-hp;
                                if(total>=since && d[s][pos]<=since && d[s][a]<=total-since)accept(total);
                            }
                        }
                    }
                }
            }
        }
        return best;
    }
};
}
#ifdef MISSION_TRAINING_CHECK
extern "C" __declspec(dllexport) double true_mission_check(const char* memory,const char* training_map,
    int w,int h,int pos,const int* sites,int fuel,int hp,const int* starts,int count,int first){
    Roads g(w*h);
    for(int u=0;u<w*h;u++){
        g[u].fill(-1);if(training_map[u]=='#')continue;
        for(int a=0;a<8;a++){
            int xx=u+dx[a],yy=u+dy[a]*w,q=xx+dy[a]*w;
            if(training_map[xx]!='#'&&training_map[yy]!='#'&&training_map[q]!='#')g[u][a]=q;
        }
    }
    Fields d;for(int i=0;i<5;i++)d[i]=distances(g,sites[i]);
    return History(memory,w,h,pos,sites,starts,count,fuel,hp,first!=0).support(sites,d);
}
#endif
extern "C" __declspec(dllexport) void exact_votes(const char* memory,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int fuel,int hp,const int* starts,int count,int first,
    int cap,int stop,double* out,int* audit){
    fill(out,out+6*8*3,0.);fill(audit,audit+6,0);
    Morphology sampler(memory,w,h,sites);History history(memory,w,h,pos,sites,starts,count,fuel,hp,first!=0);
    vector<array<double,24>> predictions;vector<array<double,6>> logweights;mt19937 rng(8141993);
    for(int k=0;k<cap;k++){
        audit[0]++;Roads g;if(!sampler.sample(k%4,rng,g))continue;audit[1]++;
        Fields d;
        // Goal distances are only required for the mission constraint and downstream value.
        for(int s=1;s<5;s++)d[s]=distances(g,sites[s]);
        auto nearest=[&](int kind,int p){return min(d[1+2*kind][p],d[2+2*kind][p]);};
        int urg=hp-now[1]<fuel-now[0],absdiff=0,delta=0;double error=0.;
        for(int s=0;s<2;s++){
            int ds=abs(nearest(s,pos)-now[s]);
            int grad=old[s]>=0?abs((nearest(s,prior)-nearest(s,pos))-(old[s]-now[s])):0;
            absdiff+=ds;delta+=grad;error+=(s==urg?.60:.32)*ds+(s==urg?2.9:1.5)*grad;
        }
        audit[2]+=delta==0;audit[3]+=absdiff==0&&delta==0;
        if(error>13)continue;
        d[0]=distances(g,sites[0]);
        double witness=history.support(sites,d);
        audit[4]+=witness>0;audit[5]+=witness>0&&absdiff==0&&delta==0;
        double logmission=witness>0?log(witness):-1000.;
        logweights.push_back({-error,-error+.6*log(max(witness,.017)),
            -error+logmission,(delta==0&&absdiff==0)?0.:-1000.,
            (delta==0&&absdiff==0)?log(max(.013,witness)):-1000.,
            (delta==0&&absdiff==0)?logmission:-1000.});
        array<double,24> v{};
        for(int a=0;a<8;a++)if(g[pos][a]>=0){
            int q=g[pos][a];
            v[3*a]=nearest(0,q)<nearest(0,pos);v[3*a+1]=nearest(1,q)<nearest(1,pos);
            int f=q==sites[1]||q==sites[2]?100:fuel-1,h=q==sites[3]||q==sites[4]?100:hp-1;
            v[3*a+2]=q==sites[0]||(min(f,h)>0 && route(q,f,h,sites,d)<inf);
        }
        predictions.push_back(v);
        if(audit[5]>=stop && k%4==3)break;
    }
    for(int j=0;j<6;j++){
        double peak=-1.e20,mass=0.;
        for(auto v:logweights)peak=max(peak,v[j]);
        if(peak<-500.)continue;
        for(size_t i=0;i<predictions.size();i++){
            double weight=exp(logweights[i][j]-peak);mass+=weight;
            for(int v=0;v<24;v++)out[24*j+v]+=weight*predictions[i][v];
        }
        for(int v=0;v<24;v++)out[24*j+v]/=mass;
    }
}
