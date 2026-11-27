// Plan several physical steps on one sample of synthetic roads and value the plan on another.
#include "morphology_worlds.cpp"
#include <memory>
#include <unordered_map>
namespace {
struct Scenario {
    Roads roads;Fields distance;
    double logw,weight;int f,h,urgent,first_d,fold;
    int near(int kind,int p)const{return min(distance[1+2*kind][p],distance[2+2*kind][p]);}
};
struct Sample {int id,f,h;bool touched;};
struct Node {int action=-1;unordered_map<int,unique_ptr<Node>> children;};
struct Decision {
    int w;const int* sites;vector<Scenario> scenarios;
    int signature(const Scenario& m,int p,int before)const{
        int mask=0;
        for(int a=0;a<8;a++)if(m.roads[p][a]>=0)mask|=1<<a;
        return mask+256*((1+m.near(0,p)-m.near(0,before))+3*(1+m.near(1,p)-m.near(1,before)));
    }
    Sample step(Sample s,int next)const{
        s.f=(next==sites[1]||next==sites[2])?100:s.f-1;
        s.h=(next==sites[3]||next==sites[4])?100:s.h-1;
        int u=scenarios[s.id].urgent;
        s.touched|=next==sites[1+2*u]||next==sites[2+2*u];return s;
    }
    bool ended(Sample s,int p,int depth,int horizon)const{
        return s.touched||p==sites[0]||min(s.f,s.h)<=0||depth>=horizon;
    }
    double utility(Sample s,int p,int depth)const{
        if(p==sites[0])return 1.6;
        if(min(s.f,s.h)<=0)return 0.;
        auto& m=scenarios[s.id];int urgent=m.urgent,dist=m.near(urgent,p),stock=urgent?s.h:s.f;
        bool intact=s.touched||dist<=stock||m.distance[0][p]<=stock;
        double progress=s.touched?1.:clamp((m.first_d-dist+double(depth))/(2*depth),0.,1.);
        int route_left=route(p,s.f,s.h,sites,m.distance);
        return .76*intact+.36*(route_left<inf)+.30*progress+.08*s.touched;
    }
    int fallback(Sample s,int p,int before)const{
        auto& m=scenarios[s.id];
        // Fallback sees the legal next moves, both current and preceding readings, and service markers.
        int kind=(s.h-m.near(1,p)<s.f-m.near(0,p));
        int distance=m.near(kind,p),previous=m.near(kind,before);
        double best=-1.e10;int action=-1;
        auto spatial=[&](int cell,int site){
            int x=abs(cell%w-site%w),y=abs(cell/w-site/w);
            return max(x,y)+.35*min(x,y);
        };
        for(int a=0;a<8;a++){
            int q=m.roads[p][a];if(q<0)continue;
            double score=0.;
            double target=1.e9;
            for(int i=1+2*kind;i<=2+2*kind;i++){
                double from=spatial(p,sites[i]);
                if(max(abs(p%w-sites[i]%w),abs(p/w-sites[i]/w))>distance)continue;
                target=min(target,spatial(q,sites[i])+.1*from);
            }
            if(target<1.e8)score-=target;
            if(q==before)score+=(previous<distance?4.:(previous>distance?-7.:-1.));
            if(q==sites[1+2*kind]||q==sites[2+2*kind])score+=100.;
            if(score>best){action=a;best=score;}
        }
        return action;
    }
    double apply(const Node* node,Sample s,int p,int before,int depth,int horizon)const{
        auto& m=scenarios[s.id];
        while(!ended(s,p,depth,horizon)){
            int a=node&&node->action>=0?node->action:fallback(s,p,before);
            int next=m.roads[p][a];
            if(next<0)return 0.;
            s=step(s,next);depth++;
            if(node){
                auto it=node->children.find(signature(m,next,p));
                node=it==node->children.end()?nullptr:it->second.get();
            }
            before=p;p=next;
        }
        return utility(s,p,depth);
    }
    pair<double,unique_ptr<Node>> optimize(const vector<Sample>& group,int p,int before,int depth,int horizon)const{
        double total=0.,squares=0.;int count=0;auto node=make_unique<Node>();
        for(auto s:group){double mass=scenarios[s.id].weight;total+=mass;squares+=mass*mass;count+=!ended(s,p,depth,horizon);}
        if(!count||total*total<3.5*squares){
            double value=0.;
            for(auto s:group)value+=scenarios[s.id].weight*apply(nullptr,s,p,before,depth,horizon);
            return {value,move(node)};
        }
        auto& first=scenarios[group.front().id];double best=0.;
        for(auto s:group)best+=scenarios[s.id].weight*apply(nullptr,s,p,before,depth,horizon);
        vector<pair<double,int>> actions;
        for(int a=0;a<8;a++)if(first.roads[p][a]>=0)actions.push_back({0.,a});
        if(actions.size()>3 && depth+1<horizon){
            for(auto& item:actions)for(auto s:group){
                if(ended(s,p,depth,horizon))continue;
                int next=first.roads[p][item.second];
                item.first+=scenarios[s.id].weight*(utility(step(s,next),next,depth+1)
                                      +.035*(fallback(s,p,before)==item.second));
            }
            stable_sort(actions.begin(),actions.end(),[](auto lhs,auto rhs){return lhs.first>rhs.first;});
            actions.resize(3);
        }
        for(auto item:actions){
            int a=item.second,next=first.roads[p][a];
            unordered_map<int,vector<Sample>> buckets;double score=0.;
            for(auto s:group){
                auto& m=scenarios[s.id];
                if(ended(s,p,depth,horizon)){
                    score+=m.weight*utility(s,p,depth);continue;
                }
                Sample ns=step(s,next);
                if(ended(ns,next,depth+1,horizon))score+=m.weight*utility(ns,next,depth+1);
                else buckets[signature(m,next,p)].push_back(ns);
            }
            unordered_map<int,unique_ptr<Node>> children;
            for(auto& b:buckets){
                auto res=optimize(b.second,next,p,depth+1,horizon);score+=res.first;
                children.emplace(b.first,move(res.second));
            }
            if(score>best){best=score;node->action=a;node->children=move(children);}
        }
        return {best,move(node)};
    }
    void evaluate(int pos,int horizon,double* output)const{
        double mass=0.;for(auto& m:scenarios)mass+=m.weight;
        for(int a=0;a<8;a++){
            int p=scenarios.front().roads[pos][a];if(p<0)continue;
            for(int fold=0;fold<2;fold++){
                unordered_map<int,vector<Sample>> train,test;
                for(int j=0;j<int(scenarios.size());j++){
                    auto& m=scenarios[j];Sample s=step({j,m.f,m.h,false},p);
                    auto& dest=(m.fold==fold?train:test);
                    dest[signature(m,p,pos)].push_back(s);
                }
                for(auto& b:test){
                    unique_ptr<Node> plan;
                    auto it=train.find(b.first);
                    if(it!=train.end())plan=optimize(it->second,p,pos,1,horizon).second;
                    for(auto s:b.second){
                        double weight=scenarios[s.id].weight/mass;
                        output[a*6]+=weight*apply(plan.get(),s,p,pos,1,horizon);
                        output[a*6+1]+=weight*apply(nullptr,s,p,pos,1,horizon);
                    }
                }
            }
        }
    }
};
}
extern "C" __declspec(dllexport) void observable_tree(const char* memory,int w,int h,int pos,int prior,
    const int* sites,const int* now,const int* old,int fuel,int hp,const int* starts,int nstarts,int initial,int particles,double* out){
    fill(out,out+8*6,0.);
    Morphology sampler(memory,w,h,sites);mt19937 rng(419396);
    Decision search{w,sites,{}};int k=hp-now[1]<fuel-now[0];
    for(int i=0;i<particles;i++){
        Scenario s;if(!sampler.sample(i%4,rng,s.roads))continue;
        for(int j=0;j<5;j++)s.distance[j]=distances(s.roads,sites[j]);
        int nf=s.near(0,pos),nr=s.near(1,pos);
        if(max(nf,nr)>=inf)continue;
        double err=0.;
        for(int j=0;j<2;j++){
            int d=s.near(j,pos);
            err+=(j==k?.6:.32)*abs(d-now[j]);
            if(old[j]>=0)err+=(j==k?2.9:1.5)*abs((s.near(j,prior)-d)-(old[j]-now[j]));
        }
        s.logw=-err-.8*witness(w,h,sites,s.distance,fuel,hp,starts,nstarts,initial!=0);
        s.f=clamp(fuel+nf-now[0],1,100);s.h=clamp(hp+nr-now[1],1,100);
        s.urgent=k;s.first_d=s.near(k,pos);s.fold=(i/4)%2;
        search.scenarios.push_back(move(s));
    }
    if(search.scenarios.empty())return;
    double peak=-1.e9;for(auto& s:search.scenarios)peak=max(peak,s.logw);
    for(auto& s:search.scenarios)s.weight=exp(s.logw-peak);
    search.evaluate(pos,3,out);search.evaluate(pos,5,out+2);search.evaluate(pos,7,out+4);
}
