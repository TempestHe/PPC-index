#pragma once
#include <string.h>
#include <cmath>
#include "config.hpp"
#include "graph.hpp"
#include "embedding.hpp"


bool check_nlf(Graph* query, int query_anchor, Graph* data, int data_anchor){
    unordered_map<Vertex, Vertex>& query_dist = query->nlf_data[query_anchor];
    unordered_map<Vertex, Vertex>& data_dist = data->nlf_data[data_anchor];
    for(auto p : query_dist){
        auto itf = data_dist.find(p.first);
        if(itf == data_dist.end()){
            return false;
        }
        if(p.second > itf->second){
            return false;
        }
    }
    return true;
}

class tree_node{
public:
    vector<Vertex> candidate;
    vector<float> candidate_score;
    vector<vector<pair<vector<Vertex>, vector<float>>>> neighbor_candidates; // query_id, data_offset,pair<neighbor id, score>
};

bool cmp_score(const pair<Vertex, float>& f1, const pair<Vertex, float>& f2){
    return f1.second > f2.second;
}

inline void sort_candidates_by_id(vector<Vertex>& candidate, vector<float>& score){
    int size = candidate.size();
    for(int i=0;i<size-1;++i){
        for(int j=i+1;j<size;++j){
            if(candidate[i]>candidate[j]){
                swap(candidate[i],candidate[j]);
                swap(score[i],score[j]);
            }
        }
    }
}

inline void sort_candidates_by_score(vector<Vertex>& candidate, vector<float>& score){
    int size = candidate.size();
    for(int i=0;i<size-1;++i){
        for(int j=i+1;j<size;++j){
            if(score[i]<score[j]){
                swap(candidate[i],candidate[j]);
                swap(score[i],score[j]);
            }
        }
    }
}

class candidate_auxiliary{
private:
    Graph* data_graph;
    Graph* query_graph;
    Tensor* query_vertex_emb;
    Tensor* data_vertex_emb;
    Tensor* query_edge_emb;
    Tensor* data_edge_emb;

    vector<vector<Vertex>> candidate_query;
    bool** candidate_bit_map;
    bool* flag;
    bool enable_order;

    vector<Vertex> edges_to_prune;
    
public:
    Vertex query_vertex_count, data_vertex_count;
    vector<Vertex> order;
    tree_node* candidates;

    // used for comparison between edge_embeddings
    vector<unordered_map<Vertex, Vertex>>* query_edge_id_map;
    vector<unordered_map<Vertex, Vertex>>* data_edge_id_map;
    Value** query_edge_emb_content, ** data_edge_emb_content;
    int edge_emb_column_size;
    // used for comparison between vertex_embeddings
    Value** query_vertex_emb_content, ** data_vertex_emb_content;
    int vertex_emb_column_size;

    candidate_auxiliary(Graph* data_graph_, Graph* query_graph_, Tensor* query_vertex_emb_, Tensor* data_vertex_emb_, Tensor* query_edge_emb_, Tensor* data_edge_emb_, bool enable_order_){
        data_graph = data_graph_;
        query_graph = query_graph_;
        query_vertex_emb = query_vertex_emb_;
        data_vertex_emb = data_vertex_emb_;
        query_edge_emb = query_edge_emb_;
        data_edge_emb = data_edge_emb_;
        enable_order = enable_order_;
        if(query_edge_emb != NULL){
            data_graph->set_up_edge_id_map();
            query_graph->set_up_edge_id_map();
            data_edge_id_map = &(data_graph->edge_id_map);
            query_edge_id_map = &(query_graph->edge_id_map);
            data_edge_emb_content = data_edge_emb->content;
            query_edge_emb_content = query_edge_emb->content;
            edge_emb_column_size = data_edge_emb->column_size;
        }
        if(query_vertex_emb != NULL){
            data_vertex_emb_content = data_vertex_emb->content;
            query_vertex_emb_content = query_vertex_emb->content;
            vertex_emb_column_size = data_vertex_emb->column_size;
        }
        if(query_vertex_emb==NULL && query_edge_emb == NULL){
            enable_order = false;
        }
        construct();
    }

    void construct(){
        query_vertex_count = query_graph->label_map.size();
        data_vertex_count = data_graph->label_map.size();
        flag = new bool [data_vertex_count];
        memset(flag, 0, sizeof(bool)*data_vertex_count);

        candidate_bit_map = new bool* [query_vertex_count];
        for(Vertex u=0;u<query_vertex_count;++u){
            candidate_bit_map[u] = new bool [data_vertex_count];
            memset(candidate_bit_map[u], 0, sizeof(bool)*data_vertex_count);
        }

        LDF_NLF();
        prune();
        link();
    }

    // bool debug_vertex = true;
    // bool debug_edge = true;
    void LDF_NLF(){
        query_graph->build_nlf();
        data_graph->build_nlf();
        vector<Label>& query_label_map = query_graph->label_map;
        vector<Label>& data_label_map = data_graph->label_map;
        for(Vertex v=0;v<data_vertex_count;++v){
            for(Vertex u=0;u<query_vertex_count;++u){
                if(query_label_map[u] == data_label_map[v] && check_nlf(query_graph, u, data_graph, v)){
                    if(query_vertex_emb != NULL){
                        if(validate_vertex_emb(u, v)==true){
                            flag[v] = true;
                            candidate_bit_map[u][v] = true;
                        }
                        // else{
                        //     if(debug_vertex){
                        //         cout<<"vertex pruned"<<endl;
                        //         debug_vertex = false;
                        //     }
                        // }
                        // if(query_emb_vertex->get_row(u)->is_contained(data_emb_vertex->get_row(v))){
                        //     flag[v] = true;
                        //     candidate_bit_map[u][v] = true;
                        // }
                    }else{
                        flag[v] = true;
                        candidate_bit_map[u][v] = true;
                    }
                }
            }
        }
    }

    double calc_score(Value* query_vec, Value* data_vec, int column_size){
        double result = 0.0;
        double sum = 0.0;
        for(int i=0;i<column_size;++i){
            assert(query_vec[i] <= data_vec[i]);
            result += (data_vec[i]-query_vec[i])*query_vec[i];
            // result += tanh(data_vec[i]-query_vec[i])*query_vec[i];
            sum += query_vec[i];
        }
        return result/sum;
    }

    void link(){
        Label max_label = *max_element(query_graph->label_map.begin(), query_graph->label_map.end());
        candidates = new tree_node [query_vertex_count];

        generate_order_in();
        Vertex** candidate_offset_map = new Vertex* [query_vertex_count];
        float* score = new float [data_vertex_count];
        for(int i=order.size()-1; i>=0; --i){
            // initialize
            Vertex u = order[i];
            Vertex offset = 0;
            candidate_offset_map[u] = new Vertex [data_vertex_count];
            memset(candidate_offset_map[u], 0, sizeof(Vertex)*data_vertex_count);

            vector<pair<Vertex, float>> candidates_score;
            
            vector<Vertex>& cans = candidates[u].candidate;
            vector<float>& can_score = candidates[u].candidate_score;
            for(Vertex v=0;v<data_vertex_count;++v){
                if(candidate_bit_map[u][v] == true){
                    if(query_vertex_emb != NULL && enable_order == true){
                        candidates_score.push_back({v, calc_score(query_vertex_emb_content[u], data_vertex_emb_content[v], vertex_emb_column_size)});
                    }else{
                        cans.push_back(v);
                        candidate_offset_map[u][v] = offset;
                        offset ++;
                    }
                    
                }
            }
            if(query_vertex_emb != NULL && enable_order == true){
                sort(candidates_score.begin(), candidates_score.end(), cmp_score);
                for(auto p : candidates_score){
                    cans.push_back(p.first);
                    can_score.push_back(p.second);
                    candidate_offset_map[u][p.first] = offset;
                    offset ++;
                }
                can_score.shrink_to_fit();
            }

            candidates[u].candidate.shrink_to_fit();
        }

        bool enable_edge_order = enable_order && query_edge_emb != NULL;

        // linkup
        for(Vertex i=0;i<order.size(); ++i){
            Vertex u = order[i];
            candidates[u].neighbor_candidates.resize(query_vertex_count);
            for(Vertex j=i+1;j<order.size();++j){
                Vertex u_n = order[j];

                if(query_graph->adj[u].find(u_n) != query_graph->adj[u].end()){
                    candidates[u].neighbor_candidates[u_n].resize(candidates[u].candidate.size());
                    for(int x=0;x<candidates[u].candidate.size();++x){
                        Vertex v = candidates[u].candidate[x];
                        vector<Vertex>& cans = candidates[u].neighbor_candidates[u_n][x].first;
                        // vector<float>& cans_score = candidates[u].neighbor_candidates[u_n][x].second;
                        if(query_edge_emb != NULL){
                            Vertex q_e_id = u<u_n ? (*query_edge_id_map)[u][u_n] : (*query_edge_id_map)[u_n][u];
                            for(auto v_n : data_graph->adj[v]){    
                                Vertex d_e_id = v<v_n ? (*data_edge_id_map)[v][v_n] : (*data_edge_id_map)[v_n][v];
                                if(candidate_bit_map[u_n][v_n] == true && validate_edge_emb(u, u_n, v, v_n) == true){
                                    cans.push_back(candidate_offset_map[u_n][v_n]);
                                    // if(enable_order){ // the edge order is enabled, we store the score along with the candidates
                                    //     cans_score.push_back(calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
                                    // }
                                }
                            }
                            cans.shrink_to_fit();
                            // cans_score.shrink_to_fit();
                            // if(enable_order == true){
                            //     sort_candidates_by_id(cans, cans_score);
                            // }else{
                                sort(cans.begin(), cans.end());
                            // }
                        }else{
                            for(auto v_n : data_graph->adj[v]){
                                if(candidate_bit_map[u_n][v_n] == true){
                                    cans.push_back(candidate_offset_map[u_n][v_n]);
                                }
                            }
                            cans.shrink_to_fit();
                            sort(cans.begin(), cans.end());
                        }
                    }
                }
            }
        }

        for(int i=0;i<query_vertex_count;++i){
            delete [] candidate_offset_map[i];
        }
        delete [] candidate_offset_map;
        delete [] score;
    }

    void print(){
        for(int i=0;i<order.size();++i){
            Vertex u = order[i];
            cout<<"======== query id:"<<u<<endl;
            cout<<"candidates:";
            for(auto v : candidates[u].candidate){
                cout<<v<<", ";
            }
            cout<<endl;

            for(Vertex j=i+1;j<order.size();++j){
                Vertex u_n = order[j];
                if(query_graph->adj[u].find(u_n) != query_graph->adj[u].end()){
                    cout<<"---------------------"<<endl;
                    cout<<"query_neighbor:"<<u_n<<endl;
                    for(int x=0;x<candidates[u].candidate.size();++x){
                        cout<<"data candidate:"<<candidates[u].candidate[x]<<endl;
                        for(auto c : candidates[u].neighbor_candidates[u_n][x].first){
                            cout<<c<<", ";
                        }
                        cout<<endl;
                    }

                }
            }

        }
    }

    inline bool validate_edge_emb(Vertex& u, Vertex& u_, Vertex& v, Vertex& v_){
        Vertex e_q_id = u<u_ ? (*query_edge_id_map)[u][u_] : (*query_edge_id_map)[u_][u];
        Vertex e_d_id = v<v_ ? (*data_edge_id_map)[v][v_] : (*data_edge_id_map)[v_][v];
        
        return vec_validation(query_edge_emb_content[e_q_id], data_edge_emb_content[e_d_id], edge_emb_column_size);
    }

    inline bool validate_vertex_emb(Vertex& u, Vertex& v){
        return vec_validation(query_vertex_emb_content[u], data_vertex_emb_content[v], vertex_emb_column_size);
    }

    void prune(){
        // first scan
        unordered_set<Vertex> to_check;
        for(Vertex v=0;v<data_vertex_count;++v){
            to_check.insert(v);
        }
        
        while(to_check.size() > 0){
            // start filtering
            unordered_set<Vertex> to_check_next;
            for(auto n : to_check){
                for(Vertex u=0; u<query_vertex_count;++u){
                    if(candidate_bit_map[u][n] == true){
                        bool whether_filter = false;
                        for(auto u_n : query_graph->adj[u]){
                            bool pass = false;
                            for(auto v_n : data_graph->adj[n]){
                                if(candidate_bit_map[u_n][v_n] == true){
                                    if(query_edge_emb != NULL &&validate_edge_emb(u, u_n, n, v_n)==false){
                                        continue;
                                    }
                                    pass = true;
                                    break;
                                }
                            }
                            if(pass == false){
                                whether_filter = true;
                                break;
                            }
                        }
                        if(whether_filter == true){
                            candidate_bit_map[u][n] = false;
                            for(auto n1 : data_graph->adj[n]){
                                if(flag[n1] == true){
                                    to_check_next.insert(n1);
                                }
                            }
                        }
                    }
                }
            }
            swap(to_check, to_check_next);
        }
    }

    void get_candidate_info(uint32_t& candidate_vertices, uint32_t& candidate_edges){
        candidate_vertices = 0;
        candidate_edges = 0;
        for(Vertex u=0;u<query_graph->label_map.size();++u){
            candidate_vertices += candidates[u].candidate.size();
        }
        for(Vertex i=0;i<order.size();++i){
            Vertex u = order[i];
            for(Vertex j=i+1;j<order.size();++j){
                Vertex u_n = order[j];
                if(query_graph->adj[u].find(u_n) != query_graph->adj[u].end()){
                    for(int x=0;x<candidates[u].candidate.size();++x){
                        candidate_edges += candidates[u].neighbor_candidates[u_n][x].first.size();
                    }
                }
            }
        }
    }

// GraphQL style
    void generate_order_in(){
        if(!order.empty()){
            return;
        }
        vector<pair<Vertex, int> > candidates_size;
        for(Vertex u=0; u<query_graph->label_map.size(); ++u){
            int size = 0;
            for(Vertex v=0;v<data_vertex_count-1;++v){
                if(candidate_bit_map[u][v] == true){
                    size ++;
                }
            }
            candidates_size.push_back({u, size});
        }
        // reorder
        for(int i=candidates_size.size();i>0;--i){
            for(int j=0;j<i-1;++j){
                if(candidates_size[j].second>candidates_size[j+1].second){
                    swap(candidates_size[j], candidates_size[j+1]);
                }else if(candidates_size[j].second == candidates_size[j+1].second){
                    Vertex d1 = candidates_size[j].first;
                    Vertex d2 = candidates_size[j+1].first;
                    if(query_graph->adj[d1].size() < query_graph->adj[d2].size()){
                        swap(candidates_size[j], candidates_size[j+1]);
                    }
                }
            }
        }
        unordered_set<Vertex> next_candidates = query_graph->adj[candidates_size[0].first];
        unordered_set<Vertex> added_candidates = {candidates_size[0].first};
        order.push_back(candidates_size[0].first);
        for(int i=1;i<query_graph->label_map.size();++i){
            for(int j=0;j<query_graph->label_map.size();++j){
                if(added_candidates.find(candidates_size[j].first) != added_candidates.end()){
                    continue;
                }
                if(next_candidates.find(candidates_size[j].first) != next_candidates.end()){
                    order.push_back(candidates_size[j].first);
                    for(auto u:query_graph->adj[candidates_size[j].first]){
                        next_candidates.insert(u);
                    }
                    added_candidates.insert(candidates_size[j].first);
                    break;
                }
            }
        }
    }

    void generate_search_order(vector<Vertex>& search_order, vector<vector<Vertex>>& pre){
        generate_order_in();
        search_order = order;
        int query_vertex_count = query_graph->label_map.size();
        pre.resize(query_vertex_count);
        for(int i=0;i<query_vertex_count;++i){
            for(int j=0;j<i;++j){
                if(query_graph->adj[order[i]].find(order[j]) != query_graph->adj[order[i]].end()){
                    pre[order[i]].push_back(order[j]);
                }
            }
        }


        if(query_edge_emb != NULL && enable_order){
            for(Vertex i=1;i<search_order.size();++i){
                Vertex u = search_order[i];
                Vertex last_parent = *(pre[search_order[i]].rbegin());

                vector<float>& score_u = candidates[u].candidate_score;

                Vertex q_e_id = last_parent<u ? (*query_edge_id_map)[last_parent][u] : (*query_edge_id_map)[u][last_parent];
                for(int y=0;y<candidates[last_parent].candidate.size();++y){
                    Vertex c = candidates[last_parent].candidate[y];
                    vector<float>& edge_score = candidates[last_parent].neighbor_candidates[u][y].second;
                    edge_score.reserve(candidates[last_parent].neighbor_candidates[u][y].first.size());
                    for(Vertex c_n_offset : candidates[last_parent].neighbor_candidates[u][y].first){
                        Vertex c_n = candidates[u].candidate[c_n_offset];
                        Vertex d_e_id = c_n<c ? (*data_edge_id_map)[c_n][c] : (*data_edge_id_map)[c][c_n];
                        edge_score.push_back(calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
                        // edge_score.push_back(score_u[c_n_offset]+calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
                    }
                }
            }
        }
        // if(query_edge_emb != NULL && enable_order){
        //     for(Vertex i=1;i<search_order.size();++i){
        //         Vertex u = search_order[i];
        //         Vertex last_parent = *(pre[search_order[i]].rbegin());

        //         vector<float>& score_u = candidates[u].candidate_score;
        //         for(int x=0;x<pre[search_order[i]].size()-1;++x){
        //             Vertex parent = pre[search_order[i]][x];
        //             Vertex q_e_id = parent<u ? (*query_edge_id_map)[parent][u] : (*query_edge_id_map)[u][parent];
        //             for(int y=0;y<candidates[parent].candidate.size();++y){
        //                 Vertex c = candidates[parent].candidate[y];
        //                 for(Vertex c_n_offset : candidates[parent].neighbor_candidates[u][y].first){
        //                     Vertex c_n = candidates[u].candidate[c_n_offset];
        //                     Vertex d_e_id = c_n<c ? (*data_edge_id_map)[c_n][c] : (*data_edge_id_map)[c][c_n];
        //                     score_u[c_n_offset] += calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size);
        //                 }
        //             }
        //         }
                
        //         Vertex q_e_id = last_parent<u ? (*query_edge_id_map)[last_parent][u] : (*query_edge_id_map)[u][last_parent];
        //         for(int y=0;y<candidates[last_parent].candidate.size();++y){
        //             Vertex c = candidates[last_parent].candidate[y];
        //             vector<float>& edge_score = candidates[last_parent].neighbor_candidates[u][y].second;
        //             edge_score.reserve(candidates[last_parent].neighbor_candidates[u][y].first.size());
        //             for(Vertex c_n_offset : candidates[last_parent].neighbor_candidates[u][y].first){
        //                 Vertex c_n = candidates[u].candidate[c_n_offset];
        //                 Vertex d_e_id = c_n<c ? (*data_edge_id_map)[c_n][c] : (*data_edge_id_map)[c][c_n];
        //                 edge_score.push_back(score_u[c_n_offset]+calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
        //             }
        //         }
        //     }
        // }
    }

    ~candidate_auxiliary(){
        delete [] candidates;
        delete [] flag;
        for(Vertex u=0;u<query_vertex_count;++u){
            delete [] candidate_bit_map[u];
        }
        delete [] candidate_bit_map;
    }
};
