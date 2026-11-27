// Commit the known-road part of an excursion before taking expectations over unknown roads.
#include "proper_frontier.cpp"
extern "C" void departure_tails(const char*,int,int,int,int,const int*,const int*,const int*,int,int,const int*,double*);
extern "C" __declspec(dllexport) void departures(const char* mem,int w,int h,int pos,int oldpos,
    const int* sites,const int* now,const int* old,int f,int hp,int cap,double* out){
    fill(out,out+8*6,0.);
    static unique_ptr<Surface> board;
    if(!board || board->memory!=mem || board->w!=w || board->h!=h || !equal(sites,sites+5,board->sites.begin()))
        board=make_unique<Surface>(mem,w,h,sites);
    auto& b=*board;
    vector<int> ds,first;shortest(b.known,pos,w,ds,first);
    vector<int> records,masks;
    for(auto c:choices(b,{f,hp},ds,first)){
        auto& dist=c.at<0?ds:b.kd[c.at];
        for(int v:b.fringe){
            int length=dist[v],mask=c.at<0?first[v]:c.mask;
            if(!mask||length>=min(c.s.fuel,c.s.hp))continue;
            records.insert(records.end(),{v,c.s.fuel-length,c.s.hp-length,c.t+length});masks.push_back(mask);
        }
    }
    if(masks.empty())return;
    vector<double> tails(masks.size()*6);
    departure_tails(mem,w,h,pos,oldpos,sites,now,old,cap,int(masks.size()),records.data(),tails.data());
    for(int j=0;j<int(masks.size());j++)for(int a=0;a<8;a++)if(masks[j]&(1<<a)){
        for(int z=0;z<6;z++){
            double value=tails[6*j+z];
            // Break equal success probabilities by time spent reaching the frontier.
            if(z<3)value*=exp(-records[4*j+3]/400.);
            out[6*a+z]=max(out[6*a+z],value);
        }
    }
}
