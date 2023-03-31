#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>

#include "embedding.hpp"
#include "config.hpp"
#include "utils.hpp"

using namespace std;


class Graph{
public:
    vector<Label> label_map;
    vector<unordered_set<Vertex>> adj;
    vector<unordered_map<Vertex, Vertex>> nlf_data;

    vector<unordered_map<Vertex, Vertex>> edge_id_map;
    vector<Vertex> estimated_common_neighbor_count;
    vector<Vertex> edge_set;
    Vertex max_label;
    Vertex edge_count = 0;

    Vertex** common_edge_neighbor; // e_id -> list of neighbors

    Graph(vector<vector<Vertex> >& adj_, vector<Label>& label_map_){
        for(auto it=adj_.begin(); it!=adj_.end(); it++){
            adj.push_back({});
            unordered_set<Vertex>& tail = adj.back();
            copy(it->begin(), it->end(), inserter(tail, tail.begin()));
        }
        common_edge_neighbor = NULL;
        copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
    }

    Graph(vector<unordered_set<Vertex> >& adj_, vector<Label>& label_map_){
        for(auto it=adj_.begin(); it!=adj_.end(); it++){
            adj.push_back({});
            unordered_set<Vertex>& tail = adj.back();
            copy(it->begin(), it->end(), inserter(tail, tail.begin()));
        }
        common_edge_neighbor = NULL;
        copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
    }

    Vertex get_edge_count(){
        if(edge_count == 0){
            for(int i=0;i<adj.size();++i){
                edge_count += adj[i].size();
            }
            edge_count = edge_count/2;
        }
        return edge_count;
    }

    void set_up_edge_id_map(){
        if(!edge_id_map.empty()){
            return;
        }
        Vertex edge_count = get_edge_count();
        edge_id_map.resize(adj.size());
        edge_set.resize(edge_count*2);
        estimated_common_neighbor_count.resize(edge_count);

        Vertex e_id = 0;
        for(Vertex v=0; v<adj.size(); ++v){
            vector<Vertex> neighbors;
            neighbors.assign(adj[v].begin(), adj[v].end());
            sort(neighbors.begin(), neighbors.end());
            for(auto e : neighbors){
                if(v < e){
                    edge_id_map[v].insert({e, e_id});
                    edge_set[e_id*2] = v;
                    edge_set[e_id*2+1] = e;
                    estimated_common_neighbor_count[e_id] = (adj[v].size() < adj[e].size()) ? adj[v].size() : adj[e].size();
                    e_id ++;
                }
            }
        }
    }

    void construct_edge_common_neighbor(){
        set_up_edge_id_map();
        if(common_edge_neighbor != NULL){
            return;
        }
        common_edge_neighbor = new Vertex* [get_edge_count()];
        for(Vertex v=0; v<adj.size(); ++v){
            for(auto n : adj[v]){
                if(v < n){
                    Vertex e_id = edge_id_map[v][n];
                    common_edge_neighbor[e_id] = new Vertex [estimated_common_neighbor_count[e_id]*3+1];
                    Vertex* vec = common_edge_neighbor[e_id];
                    int offset = 1;
                    int count = 0;
                    for(auto v_n : adj[n]){
                        if(v_n !=v && adj[v].find(v_n) != adj[v].end()){
                            vec[offset++] = v_n;
                            count++;
                            // first push_back the e_id connecting source vertex
                            if(v_n<v){
                                if(edge_id_map[v_n].find(v)==edge_id_map[v_n].end()){
                                    cout<<"Error1:"<<v_n<<" is not connected to "<<v<<endl;
                                }
                                vec[offset++] = edge_id_map[v_n][v];
                                // vec.push_back(edge_id_map[v_n][v]);
                            }else{
                                if(edge_id_map[v].find(v_n)==edge_id_map[v].end()){
                                    cout<<"Error2:"<<v<<" is not connected to "<<v_n<<endl;
                                }
                                vec[offset++] = edge_id_map[v][v_n];
                                // vec.push_back(edge_id_map[v][v_n]);
                            }
                            // second push_back the e_id connecting target vertex
                            if(v_n<n){
                                if(edge_id_map[v_n].find(n)==edge_id_map[v_n].end()){
                                    cout<<"Error3:"<<v_n<<" is not connected to "<<n<<endl;
                                }
                                vec[offset++] = edge_id_map[v_n][n];
                                // vec.push_back(edge_id_map[v_n][n]);
                            }else{
                                if(edge_id_map[n].find(v_n)==edge_id_map[n].end()){
                                    cout<<"Error4:"<<n<<" is not connected to "<<v_n<<endl;
                                }
                                vec[offset++] = edge_id_map[n][v_n];
                                // vec.push_back(edge_id_map[n][v_n]);
                            }
                        }
                    }
                    vec[0] = count;
                }
            }
        }
    }

    void rebuild(vector<vector<Vertex> >& adj_, vector<Label>& label_map_){
        adj.clear();
        label_map.clear();
        nlf_data.clear();
        for(auto it=adj_.begin(); it!=adj_.end(); it++){
            adj.push_back({});
            unordered_set<Vertex>& tail = adj.back();
            copy(it->begin(), it->end(), inserter(tail, tail.begin()));
        }
        copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
    }

    Graph(){common_edge_neighbor = NULL;};

    Graph(string filename){
        common_edge_neighbor = NULL;
        string line;
        ifstream in(filename);
        unordered_map<Vertex, Label> label_map_unordered;
        unordered_map<Vertex, unordered_set<Vertex>> adj_unordered;
        while(getline(in, line)){
            if(line.c_str()[0] == 'v'){
                stringstream ss(line);
                Vertex v, n;
                char c;
                ss>>c>>v>>n;
                label_map_unordered[v] = n;
            }else if(line.c_str()[0] == 'e'){
                stringstream ss(line);
                Vertex v, n;
                char c;
                ss>>c>>v>>n;
                if(adj_unordered.find(v) == adj_unordered.end()){
                    adj_unordered.insert({v, {n}});
                }else{
                    adj_unordered[v].insert(n);
                }
                if(adj_unordered.find(n) == adj_unordered.end()){
                    adj_unordered.insert({n, {v}});
                }else{
                    adj_unordered[n].insert(v);
                }
            }
        }
        label_map.resize(label_map_unordered.size());
        for(auto v : label_map_unordered){
            label_map[v.first] = v.second;
        }
        adj.resize(label_map_unordered.size());
        for(auto v : adj_unordered){
            adj[v.first] = v.second;
        }
    }

    void print_graph(){
        cout<<"label_map:"<<label_map.size()<<endl;
        long n=0;
        for(int i=0;i<adj.size();++i){
            n += adj[i].size();
        }
        cout<<"adj:"<<adj.size()<<":"<<n<<endl;
        n = 0;
        for(int i=0;i<nlf_data.size();++i){
            for(auto p : nlf_data[i]){
                n += p.second;
            }
        }
        cout<<"nlf:"<<nlf_data.size()<<":"<<n<<endl;
    }

    void build_nlf(){
        if(nlf_data.size() == label_map.size()){
            return;
        }
        nlf_data.resize(label_map.size());
        for(int v=0;v<label_map.size();++v){
            for(auto n:adj[v]){
                Vertex label = label_map[n];
                auto itf = nlf_data[v].find(label);
                if(itf == nlf_data[v].end()){
                    nlf_data[v].insert({label, 1});
                }else{
                    itf->second ++;
                }
            }
        }
    }

    void subiso_candidate_generate(Graph& query, vector<pair<Vertex, vector<Vertex>>>& candidates){
        // build up the nlf
        candidates.clear();
        candidates.resize(query.label_map.size());
        for(int u=0; u<query.label_map.size(); ++u){
            candidates[u].first = u;
            candidates[u].second = {};
            for(int v=0;v<label_map.size(); ++v){
                if(query.label_map[u] == label_map[v]){
                    bool can_match = true;
                    for(auto l : query.nlf_data[u]){
                        auto itf = nlf_data[v].find(l.first);
                        if(itf == nlf_data[v].end() || itf->second < l.second){
                            can_match = false;
                            break;
                        }else if(itf->second < l.second){
                            can_match = false;
                            break;
                        }
                    }
                    if(can_match){
                        candidates[u].second.push_back(v);
                    }
                }
            }
        }
        // rank the candidates
        for(int i=candidates.size();i>0;--i){
            for(int j=0;j<i-1;++j){
                if(candidates[j].second.size()>candidates[j+1].second.size()){
                    swap(candidates[j], candidates[j+1]);
                }
            }
        }
    }

    ~Graph(){
        if(common_edge_neighbor!=NULL){
            for(int i=0;i<get_edge_count();++i){
                delete [] common_edge_neighbor[i];
            }
            delete [] common_edge_neighbor;
        }
    }
};

typedef tuple<Graph*, pair<Vertex, Vertex>*, Vertex**, int> input_para;

void* compute_common_neighbor_func(void* args){
    input_para& input = *(input_para*)args;
    Graph* graph = std::get<0>(input);
    pair<Vertex, Vertex>* task = std::get<1>(input);
    Vertex** common_neighbors = std::get<2>(input);
    int id = std::get<3>(input);
    
    vector<Vertex>& edge_set = graph->edge_set;
    vector<unordered_set<Vertex>>& adj = graph->adj;
    vector<unordered_map<Vertex, Vertex>>& edge_id_map = graph->edge_id_map;

    int test_id = 0;
    for(Vertex e_id=task->first; e_id<task->second; ++e_id){
        test_id++;
        Vertex v = graph->edge_set[e_id*2];
        Vertex n = graph->edge_set[e_id*2+1];
        int common_size = 0;
        int offset = 1;
        Vertex* vec = common_neighbors[e_id];
        for(auto v_n : graph->adj[n]){ // auto v_n : adj[n]
            if(v_n!=v && graph->adj[v].find(v_n)!=graph->adj[v].end()){
                vec[offset++] = v_n;
                common_size ++;
                if(v_n<v){
                    if(graph->edge_id_map[v_n].find(v)==graph->edge_id_map[v_n].end()){
                        cout<<"t:"<<id<<"Error1:"<<v_n<<" is not connected to "<<v<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                        continue;
                    }
                    vec[offset++] = graph->edge_id_map[v_n][v];
                }else{
                    if(graph->edge_id_map[v].find(v_n)==graph->edge_id_map[v].end()){
                        cout<<"t:"<<id<<"Error2:"<<v<<" is not connected to "<<v_n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                        continue;
                    }
                    vec[offset++] = graph->edge_id_map[v][v_n];
                }
                if(v_n<n){
                    if(graph->edge_id_map[v_n].find(n)==graph->edge_id_map[v_n].end()){
                        cout<<"t:"<<id<<"Error3:"<<v_n<<" is not connected to "<<n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                        continue;
                    }
                    vec[offset++] = graph->edge_id_map[v_n][n];
                }else{
                    if(graph->edge_id_map[n].find(v_n)==graph->edge_id_map[n].end()){
                        cout<<"t:"<<id<<"Error4:"<<n<<" is not connected to "<<v_n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                        continue;
                    }
                    vec[offset++] = graph->edge_id_map[n][v_n];
                }
            }
            // if(common_size != graph->estimated_common_neighbor_count[e_id]){
            //     exit(0);
            //     cout<<"e_id:"<<e_id<<":"<<common_size<<":"<<graph->estimated_common_neighbor_count[e_id]<<endl;
            // }
            vec[0] = common_size;
        }
    }
    return NULL;
}

// compute the common edge neighbors for graph
void common_edge_neighbor_multi_threads(Graph* graph, int thread_num){
    graph->set_up_edge_id_map();
    if(graph->common_edge_neighbor != NULL){
        return;
    }
    // initialize common_edge_neighbor
    int edge_count = graph->get_edge_count();
    graph->common_edge_neighbor = new Vertex* [edge_count];
    for(Vertex e_id=0; e_id<edge_count; ++e_id){
        graph->common_edge_neighbor[e_id] = new Vertex [3*graph->estimated_common_neighbor_count[e_id]+1];
    }
    
    thread_num = (thread_num<edge_count) ? thread_num : edge_count;
    int average_edges_per_thread = edge_count/thread_num;
    int remain_edges = edge_count%thread_num;
    // initialise the task
    vector<pair<Vertex, Vertex>> tasks;
    int offset = 0;
    int acc_edges = 0;
    for(int i=0; i<thread_num; ++i){
        if(i<remain_edges){
            tasks.push_back({offset, offset+average_edges_per_thread+1});
            offset += average_edges_per_thread+1;
        }else{
            tasks.push_back({offset, offset+average_edges_per_thread});
            offset += average_edges_per_thread;
        }
    }
    assert(offset == edge_count);

    // initialize the parameters
    vector<input_para> input_parameters;
    vector<int> fd_list;
    for(int i=0;i<thread_num;++i){
        input_parameters.push_back(input_para(graph, &(tasks[i]), graph->common_edge_neighbor, i));
    }
    // start computation
    pthread_t* threads = new pthread_t [thread_num];

    for(int i=0;i<thread_num;++i){
        int res = pthread_create(&(threads[i]), NULL, compute_common_neighbor_func, (void*)&input_parameters[i]);
        if(res != 0){
            cout<<"Create thread:"<<i<<" failed for common neighbor computation"<<endl;
            exit(res);
        }
    }

    for(int i=0; i<thread_num; ++i){
        void* ret;
        pthread_join(threads[i], &ret);
    }
    delete [] threads;
}
