#pragma once
#include "subgraph_enumeration.h"

#define _GNU_SOURCE
 
#include <sys/types.h>
#include <fcntl.h>
#include <malloc.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

timer_t id;
bool stop = false;

void set_stop_flag_enum(union sigval val){
    *((bool*)val.sival_ptr) = true;
    timer_delete(id);
}

void register_timer_enum(bool* stop_flag){
    struct timespec spec;
    struct sigevent ent;
    struct itimerspec value;
    struct itimerspec get_val;

    /* Init */
    memset(&ent, 0x00, sizeof(struct sigevent));
    memset(&get_val, 0x00, sizeof(struct itimerspec));

    int test_val = 0;
    /* create a timer */
    ent.sigev_notify = SIGEV_THREAD;
    ent.sigev_notify_function = set_stop_flag_enum;
    ent.sigev_value.sival_ptr = stop_flag;
    timer_create(CLOCK_MONOTONIC, &ent, &id);

    /* start a timer */
    value.it_value.tv_sec = TIME_LIMIT;
    value.it_value.tv_nsec = 0;
    value.it_interval.tv_sec = 0;
    value.it_interval.tv_nsec = 0;
    timer_settime(id, 0, &value, NULL);
}

bool debug(int depth, Vertex* emb, vector<Vertex>& order){
    vector<Vertex> content = {0, 1417,2194,341,2766,1845,2935,2693,1551,2344,1767,2493,1209,784,1347,1873,281,845,2783,979,1942,277,2425,1575,1505,1331,167,571,1466,890,1744,3019,3078};
    for(int i=1;i<depth+1;++i){
        if(emb[order[i]] != content[i]){
            return false;
        }
    }
    return true;
}

// void subgraph_enumeration_get_candidate_info(Graph* data_graph, Graph* query_graph, 
//     Tensor* query_vertex_emb, Tensor* data_vertex_emb,
//     Tensor* query_edge_emb, Tensor* data_edge_emb,
//     uint32_t& vertex_candidate_num, uint32_t& edge_candidate_num
// ){
//     Auxiliary aux(data_graph, query_graph, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, false);
//     aux.get_candidate_info(vertex_candidate_num, edge_candidate_num);
// }

void subgraph_enumeration(Graph* data_graph, Graph* query_graph, 
    long count_limit, long& result_count, 
    double& enumeration_time, double& preprocessing_time, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    bool enable_order // whether order the candidates
){
    stop = false;
    register_timer_enum(&stop);

    struct timeval start_t, end_t;

    gettimeofday(&start_t, NULL);
    
    Auxiliary aux(data_graph, query_graph, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, enable_order);

    gettimeofday(&end_t, NULL);
    preprocessing_time = get_time(start_t, end_t);

    vector<Vertex> order;
    vector<Vertex> order_offset;
    vector<vector<Vertex>>& pre = aux.predecessor_neighbors;
    vector<vector<Vertex>>& suc = aux.successor_neighbors;
    uint32_t query_vertex_count = query_graph->label_map.size();
    uint32_t data_vertex_count = data_graph->label_map.size();
    aux.generate_search_order(order, order_offset);
    // arbitrarily insert a value
    order.insert(order.begin(), 0);
    for(int i=0;i<order_offset.size();++i){
        order_offset[i] += 1;
    }


#ifdef FAILING_SET_PRUNING
    vector<bitset<MAX_QUERY_SIZE>>& ancestors = aux.ancestors;
    // initialize the failing set for each query vertex
    vector<bitset<MAX_QUERY_SIZE>> failing_set;
    failing_set.resize(order.size()+1);
    for(int i=0; i<order.size()+1; ++i){
        failing_set[i].reset();
    }
    // intialize the parent map
    vector<vector<bitset<MAX_QUERY_SIZE>>> parent_failing_set_map;
    parent_failing_set_map.resize(query_vertex_count);
    for(int i=0;i<query_vertex_count; ++i){
        parent_failing_set_map[i].resize(query_vertex_count);
    }
    for(Vertex u=0; u<query_vertex_count; ++u){
        for(auto successor : suc[u]){
            parent_failing_set_map[u][successor] = ancestors[u];
            for(auto parent : pre[successor]){
                if(order_offset[parent]<order_offset[u]){
                    parent_failing_set_map[u][successor] |= ancestors[parent];
                }
            }
        }
    }

#endif

    cout<<"search order:{";
    for(auto v : order){
        cout<<v<<", ";
    }
    cout<<"}"<<endl;
    // cout<<"checkpoint:{";
    // for(auto v : aux.cut_check_point){
    //     cout<<v<<", ";
    // }
    // cout<<"}"<<endl;
    // cout<<"parent:{";
    // for(int i=0; i<aux.parent_check_point.size(); ++i){
    //     cout<<i<<":"<<aux.parent_check_point[i]<<", ";
    // }
    // cout<<"}"<<endl;
    // order = {9, 5, 11, 16, 18, 20, 23, 22, 24, 26, 32, 35, 1, 6, 36, 7, 10, 3, 4, 13, 14, 25, 19, 27, 28, 29, 31, 33, 34, 39, 30, 38, 2, 0, 8, 12, 17, 15, 21, 37};
    long emb_count = 0;
    
    // start enumeration
    vector<vector<vector<Vertex>>> candidates_stack, candidates_offset_map_stack;

    candidates_stack.resize(query_vertex_count+1);
    candidates_offset_map_stack.resize(query_vertex_count+1);
    
    // initialize the starting candidates
    candidates_offset_map_stack[1].push_back({});
    candidates_stack[1].push_back(aux.candidates[order[1]].candidate);
    for(int i=0;i<aux.candidates[order[1]].candidate.size(); ++i){
        candidates_offset_map_stack[1][0].push_back(i);
    }

    vector<bitset<MAX_QUERY_SIZE>> parent_map;
    parent_map.resize(query_vertex_count);
    for(Vertex u=0; u<query_vertex_count; ++u){
        for(auto p : pre[u]){
            parent_map[u].set(p);
        }
    }
    
    bool enable_edge_order = enable_order && query_edge_emb != NULL;
    Vertex* candidates_offset = new Vertex [query_vertex_count+1];
    Vertex* embedding = new Vertex [query_vertex_count];
    Vertex* embedding_offset = new Vertex [query_vertex_count];

    bool* visited_vertices = new bool [data_vertex_count];
    Vertex* visited_query_vertices = new Vertex [data_vertex_count];
    memset(visited_vertices, 0, sizeof(bool)*data_vertex_count);
    memset(candidates_offset, 0, sizeof(Vertex)*(query_vertex_count+1));

    int cur_depth = 1;
    candidates_offset[cur_depth] = 0;
    vector<unordered_set<Vertex>>& adj_map = data_graph->adj;
    tree_node* aux_content = aux.candidates;
    bool* find_matches = new bool [query_vertex_count+1];
    find_matches[query_vertex_count] = true;

    vector<float> candidate_score;
    bool empty_conflict = false;

    gettimeofday(&start_t, NULL);
    Vertex u, v;
    while(true){
        while(candidates_offset[cur_depth]<candidates_stack[cur_depth].rbegin()->size()){
            if(stop==true){
                goto EXIT;
            }
            
            int current_stack_size = candidates_stack[cur_depth].size()-1;
            find_matches[cur_depth] = false;
#ifdef FAILING_SET_PRUNING
            failing_set[cur_depth].reset();
#endif
            u = order[cur_depth];
            // whether find a full match
            if(cur_depth == query_vertex_count){
                for(int i=0;i<candidates_stack[cur_depth][current_stack_size].size(); ++i){
                    v = candidates_stack[cur_depth][current_stack_size][i];
                    if(visited_vertices[v] == false){

                        find_matches[cur_depth-1] = true;
                        find_matches[cur_depth] = true;
                        emb_count ++;
                        embedding[u] = v;
                        // for(Vertex z=1;z<=query_vertex_count; ++z){
                        //     cout<<embedding[order[z]]<<" ";
                        // }
                        // cout<<endl;
                        // exit(0);
#ifdef FAILING_SET_PRUNING
                        // belongs to embedding-class
                        failing_set[cur_depth].reset();
#endif
                        if(emb_count >= count_limit){
                            goto EXIT;
                        }
                    }else{
#ifdef FAILING_SET_PRUNING
                        // belongs to conflict-class
                        failing_set[cur_depth] = ancestors[u] | ancestors[visited_query_vertices[v]];
                        // cout<<cur_depth<<":"<<v<<":"<<visited_query_vertices[v]<<":"<<ancestors[u]<<":"<<ancestors[visited_query_vertices[v]]<<endl;
#endif
                    }
#ifdef FAILING_SET_PRUNING
                    // updating the failing set of the parent
                    if(find_matches[cur_depth] == true){
                        failing_set[cur_depth-1].reset();
                    }else if(find_matches[cur_depth-1] == false){
                        failing_set[cur_depth-1] |= failing_set[cur_depth];
                    }
#endif
                }
                candidates_offset[cur_depth] = candidates_stack[cur_depth].rbegin()->size();
            }else{
                uint32_t offset = candidates_offset[cur_depth];
                u = order[cur_depth];
                int s = candidates_stack[cur_depth].size();
                v = candidates_stack[cur_depth][s-1][offset];
                candidates_offset[cur_depth] ++;
                if(visited_vertices[v] == true){
#ifdef FAILING_SET_PRUNING
                    // belongs to conflict-class
                    failing_set[cur_depth] = ancestors[u] | ancestors[visited_query_vertices[v]];
                    // update the failing set of the parent
                    if(find_matches[cur_depth-1] == false){
                        failing_set[cur_depth-1] |= failing_set[cur_depth];
                    }
                    // cout<<cur_depth<<":"<<v<<":"<<visited_query_vertices[v]<<":"<<ancestors[u]<<":"<<ancestors[visited_query_vertices[v]]<<endl;
#endif
                    continue;
                }
                embedding[u] = v;
                // if(debug(cur_depth, embedding, order)){
                //     cout<<"depth:"<<cur_depth<<endl;
                // }
                embedding_offset[u] = candidates_offset_map_stack[cur_depth][current_stack_size][offset];
                visited_vertices[v] = true;
                visited_query_vertices[v] = u;

                // start finding the candidates for all the successors
                bool empty_candidates = false;
                int suc_offset = 0;
                for(; suc_offset<suc[u].size(); ++suc_offset){ // auto successor : suc[u]
                    Vertex successor = suc[u][suc_offset];
                    uint32_t suc_depth = order_offset[successor];
                    // if(candidates_offset_map_stack[suc_depth].empty() == false){
                    //     candidates_offset_map_stack[suc_depth].pop_back();
                    //     candidates_stack[suc_depth].pop_back();
                    // }
                    // candidates_offset[suc_depth] = 0;
                    if(candidates_offset_map_stack[suc_depth].empty()){
                        candidates_offset_map_stack[suc_depth].push_back(aux_content[u].neighbor_candidates[successor][embedding_offset[u]].first);
                        candidates_stack[suc_depth].push_back({});
                        // map the offset in aux to real vertex id
                        for(auto aux_offset : *(candidates_offset_map_stack[suc_depth].rbegin())){
                            candidates_stack[suc_depth].rbegin()->push_back(aux_content[successor].candidate[aux_offset]);
                        }
                    }else{
                        // intersect the candidates
                        vector<Vertex>& current_candidates_offset = *(candidates_offset_map_stack[suc_depth].rbegin());
                        vector<Vertex>& suc_candidates_offset = aux_content[u].neighbor_candidates[successor][embedding_offset[u]].first;
                        
                        vector<Vertex> intersected_candidates_offset, intersected_candidates;
                        for(int i=0,j=0; i<current_candidates_offset.size() && j<suc_candidates_offset.size();){
                            if(current_candidates_offset[i] == suc_candidates_offset[j]){
                                Vertex aux_offset = current_candidates_offset[i];
                                intersected_candidates_offset.push_back(aux_offset);
                                intersected_candidates.push_back(aux_content[successor].candidate[aux_offset]);
                                ++i;
                                ++j;
                            }else if(current_candidates_offset[i] > suc_candidates_offset[j]){
                                ++j;
                            }else{
                                ++i;
                            }
                        }
                        candidates_offset_map_stack[suc_depth].push_back(intersected_candidates_offset);
                        candidates_stack[suc_depth].push_back(intersected_candidates);
                    }

                    if(candidates_stack[suc_depth].rbegin()->empty()){
                        empty_candidates = true;
#ifdef FAILING_SET_PRUNING
                        // failing_set[cur_depth] |= parent_map[successor];
                        if(failing_set[cur_depth].any() == true){
                            failing_set[cur_depth] &= parent_failing_set_map[u][successor];
                        }else{
                            failing_set[cur_depth] |= parent_failing_set_map[u][successor];
                        }
                        
#endif
                    }
                }
                if(empty_candidates == true){
                    for(int i=0;i<suc_offset;++i){
                        Vertex successor = suc[u][i];
                        uint32_t suc_depth = order_offset[successor];
                        candidates_offset_map_stack[suc_depth].pop_back();
                        candidates_stack[suc_depth].pop_back();
                        candidates_offset[suc_depth] = 0;
                    }
                    visited_vertices[v] = false;
#ifdef FAILING_SET_PRUNING
                    failing_set[cur_depth-1] |= failing_set[cur_depth];
#endif
                    // empty_conflict = true;
                    // cout<<"empty set:"<<cur_depth<<endl;
                    continue;
                }
                // empty_conflict = false;
                uint32_t next_depth = cur_depth+1;
                Vertex next_u = order[next_depth];
                candidates_offset[next_depth] = 0;
                cur_depth ++;
                if(enable_edge_order && next_depth != query_vertex_count){
                    uint32_t num_candidates = candidates_stack[next_depth].rbegin()->size();
                    candidate_score.resize(num_candidates);
                    memset(&candidate_score[0], 0, sizeof(float)*num_candidates);

                    Vertex last_parent = *(pre[next_u].rbegin());
                    vector<float> score = aux_content[last_parent].neighbor_candidates[next_u][embedding_offset[last_parent]].second;
                    for(uint32_t i=0; i<num_candidates; ++i){
                        candidate_score[i] += score[embedding_offset[last_parent]];
                    }

                    // for(auto parent : pre[next_u]){
                    //     vector<float>& score = aux_content[parent].neighbor_candidates[next_u][embedding_offset[parent]].second;
                    //     for(uint32_t i=0; i<num_candidates; ++i){
                    //         candidate_score[i] += score[embedding_offset[parent]];
                    //     }
                    // }
                    // reorder the candidates
                    vector<Vertex>& candidate_offset = *(candidates_offset_map_stack[next_depth].rbegin());
                    vector<Vertex>& candidates = *(candidates_stack[next_depth].rbegin());
                    for(int i=0;i<num_candidates-1;++i){
                        for(int j=i+1;j<num_candidates;++j){
                            if(candidate_score[i]<candidate_score[j]){
                                swap(candidate_score[i],candidate_score[j]);
                                swap(candidate_offset[i], candidate_offset[j]);
                                swap(candidates[i], candidates[j]);
                            }
                        }
                    }
                }
                
            }
        }

        cur_depth --;
        if(cur_depth == 0){
            break;
        }

#ifdef FAILING_SET_PRUNING
        // update the failing_set of the parent
        if(find_matches[cur_depth] == true){
            failing_set[cur_depth-1].reset();
        }else if(find_matches[cur_depth-1]==false){
            if(failing_set[cur_depth].test(u) == false){
                failing_set[cur_depth-1] = failing_set[cur_depth];
                // exit(0);
            }else{
                failing_set[cur_depth-1] |= failing_set[cur_depth];
            }
        }
#endif

        for(auto successor : suc[order[cur_depth]]){
            uint32_t suc_depth = order_offset[successor];
            candidates_offset_map_stack[suc_depth].pop_back();
            candidates_stack[suc_depth].pop_back();
            candidates_offset[suc_depth] = 0;
        }

#ifdef FAILING_SET_PRUNING
        find_matches[cur_depth-1] |= find_matches[cur_depth];
        if(find_matches[cur_depth] == false && failing_set[cur_depth].test(order[cur_depth]) == false){
            // filtering the redundant siblings
            candidates_offset[cur_depth] = candidates_stack[cur_depth].rbegin()->size();
            failing_set[cur_depth-1] |= failing_set[cur_depth];
            // cout<<"failing set:"<<cur_depth<<endl;
            // if(cur_depth==35){
            //     // for(Vertex d=1; d<=cur_depth;++d){
            //     //     cout<<embedding[order[d]]<<" ";
            //     // }
            //     // cout<<endl;
            //     cout<<emb_count<<endl;
            // }
        }
#endif
        visited_vertices[embedding[order[cur_depth]]] = false;
    }
EXIT:
    delete [] candidates_offset;
    delete [] embedding;
    delete [] embedding_offset;
    delete [] visited_vertices;
    delete [] visited_query_vertices;
    delete [] find_matches;
    gettimeofday(&end_t, NULL);
    if(stop == false){
        timer_delete(id);
    }
    result_count = emb_count;
    enumeration_time = get_time(start_t, end_t);
}




// void subgraph_enumeration(Graph* data_graph, Graph* query_graph, 
//     long count_limit, long& result_count, 
//     double& enumeration_time, double& preprocessing_time, 
//     Tensor* query_vertex_emb, Tensor* data_vertex_emb,
//     Tensor* query_edge_emb, Tensor* data_edge_emb,
//     bool enable_order // whether order the candidates
// ){
//     stop = false;
//     register_timer_enum(&stop);

//     struct timeval start_t, end_t;

//     gettimeofday(&start_t, NULL);
    
//     Auxiliary aux(data_graph, query_graph, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, enable_order);

//     gettimeofday(&end_t, NULL);
//     preprocessing_time = get_time(start_t, end_t);

//     vector<Vertex> order;
//     vector<Vertex> order_offset;
//     vector<vector<Vertex>>& pre = aux.predecessor_neighbors;
//     vector<vector<Vertex>>& suc = aux.successor_neighbors;
//     aux.generate_search_order(order, order_offset);
//     // arbitrarily insert a value to order
//     order.insert(order.begin(), 0);

// #ifdef CORE_DECOMPOSITION
//     vector<bool>& cut_check_point = aux.cut_check_point;
//     vector<int>& parent_check_point = aux.parent_check_point;
// #endif

// #ifdef FAILING_SET_PRUNING
//     vector<bitset<MAX_QUERY_SIZE>>& ancestors = aux.ancestors;
//     // initialize the failing set for each query vertex
//     vector<bitset<MAX_QUERY_SIZE>> failing_set;
//     failing_set.resize(order.size());
//     for(int i=0; i<order.size(); ++i){
//         failing_set[i].reset();
//     }
// #endif

//     // cout<<"search order:{";
//     // for(auto v : order){
//     //     cout<<v<<", ";
//     // }
//     // cout<<"}"<<endl;
//     // cout<<"checkpoint:{";
//     // for(auto v : aux.cut_check_point){
//     //     cout<<v<<", ";
//     // }
//     // cout<<"}"<<endl;
//     // cout<<"parent:{";
//     // for(int i=0; i<aux.parent_check_point.size(); ++i){
//     //     cout<<i<<":"<<aux.parent_check_point[i]<<", ";
//     // }
//     // cout<<"}"<<endl;
//     // order = {9, 5, 11, 16, 18, 20, 23, 22, 24, 26, 32, 35, 1, 6, 36, 7, 10, 3, 4, 13, 14, 25, 19, 27, 28, 29, 31, 33, 34, 39, 30, 38, 2, 0, 8, 12, 17, 15, 21, 37};
//     long emb_count = 0;
    
//     // start enumeration
//     vector<vector<Vertex>> candidates;
//     vector<vector<Vertex>> candidates_offset_map;
//     uint32_t query_vertex_count = query_graph->label_map.size();
//     uint32_t data_vertex_count = data_graph->label_map.size();
//     candidates.resize(query_vertex_count+1);
//     candidates_offset_map.resize(query_vertex_count+1);
    
//     // initialize the starting candidates
//     for(int i=0;i<aux.candidates[order[1]].candidate.size(); ++i){
//         candidates[1].push_back(aux.candidates[order[1]].candidate[i]);
//         candidates_offset_map[1].push_back(i);
//     }
    
//     bool enable_edge_order = enable_order && query_edge_emb != NULL;
//     Vertex* candidates_offset = new Vertex [query_vertex_count+1];
//     Vertex* embedding = new Vertex [query_vertex_count];
//     Vertex* embedding_offset = new Vertex [query_vertex_count];

//     bool* visited_vertices = new bool [data_vertex_count];
//     Vertex* visited_query_vertices = new Vertex [data_vertex_count];
//     memset(visited_vertices, 0, sizeof(bool)*data_vertex_count);

//     int cur_depth = 1;
//     candidates_offset[cur_depth] = 0;
//     vector<unordered_set<Vertex>>& adj_map = data_graph->adj;
//     tree_node* aux_content = aux.candidates;
//     bool* find_matches = new bool [query_vertex_count+1];
//     find_matches[query_vertex_count] = true;

//     vector<float> candidate_score;

//     gettimeofday(&start_t, NULL);
//     Vertex u, v;
//     while(true){
//         while(candidates_offset[cur_depth]<candidates[cur_depth].size()){
//             if(stop==true){
//                 goto EXIT;
//             }

//             find_matches[cur_depth] = false;
//             u = order[cur_depth];
//             // whether find a full match
//             if(cur_depth == query_vertex_count){
//                 for(int i=0;i<candidates[cur_depth].size(); ++i){
//                     v = candidates[cur_depth][i];
//                     if(visited_vertices[v] == false){
//                         find_matches[cur_depth-1] = true;
//                         find_matches[cur_depth] = true;
//                         emb_count ++;
//                         embedding[u] = v;
//                         for(int m=1;m<=query_vertex_count;++m){
//                             cout<<embedding[order[m]]<<" ";
//                         }
//                         cout<<endl;
//                         // exit(0);
// #ifdef FAILING_SET_PRUNING
//                         // belongs to embedding-class
//                         failing_set[cur_depth].reset();
// #endif
//                         if(emb_count >= count_limit){
//                             goto EXIT;
//                         }
//                     }else{
// #ifdef FAILING_SET_PRUNING
//                         // belongs to conflict-class
//                         failing_set[cur_depth] = ancestors[u] | ancestors[visited_query_vertices[v]];
// #endif
//                     }
// #ifdef FAILING_SET_PRUNING
//                     // updating the failing set of the parent
//                     if(find_matches[cur_depth] == true){
//                         failing_set[cur_depth-1].reset();
//                     }else if(find_matches[cur_depth-1] == false){
//                         failing_set[cur_depth-1] |= failing_set[cur_depth];
//                     }
// #endif
//                 }
//                 candidates_offset[cur_depth] = candidates[cur_depth].size();
//             }else{
//                 uint32_t offset = candidates_offset[cur_depth];
//                 u = order[cur_depth];
//                 v = candidates[cur_depth][offset];
//                 candidates_offset[cur_depth] ++;
//                 if(visited_vertices[v] == true){
// #ifdef FAILING_SET_PRUNING
//                     // belongs to conflict-class
//                     failing_set[cur_depth] = ancestors[u] | ancestors[visited_query_vertices[v]];
//                     // update the parent
//                     if(find_matches[cur_depth-1] == false){
//                         failing_set[cur_depth-1] |= failing_set[cur_depth];
//                     }else{
//                         failing_set[cur_depth-1].reset();
//                     }
// #endif
//                     continue;
//                 }
//                 embedding[u] = v;
//                 // if(debug(cur_depth, embedding, order)){
//                 //     cout<<"depth:"<<cur_depth<<endl;
//                 // }
//                 // if(debug(cur_depth+1, order, embedding)){
//                 //     cout<<"depth:"<<cur_depth<<endl;
//                 // }
//                 embedding_offset[u] = candidates_offset_map[cur_depth][offset];
//                 visited_vertices[v] = true;
//                 visited_query_vertices[v] = u;

//                 // start finding the candidates
//                 int next_depth = cur_depth+1;
//                 vector<pair<Vertex*, int>> can;
//                 Vertex max_start_id = 0;
//                 Vertex next_u = order[next_depth];
//                 for(auto parent : pre[next_u]){
//                     vector<Vertex>& tmp = aux_content[parent].neighbor_candidates[next_u][embedding_offset[parent]].first;
//                     can.push_back({&(tmp[0]), tmp.size()});
//                 }

//                 // intersect the candiate neighbors
//                 candidates[next_depth].clear();
//                 candidates_offset_map[next_depth].clear();
//                 vector<int> ptr(can.size(), 0);
//                 uint32_t num_candidates;
//                 while(true){
//                     RESCAN:
//                     for(int j=0;j<can.size();++j){
//                         int cur_ptr = ptr[j];
//                         Vertex* vec = can[j].first;
//                         int vec_size = can[j].second;
//                         while(vec[cur_ptr]<max_start_id && cur_ptr<vec_size) ++cur_ptr;
//                         ptr[j] = cur_ptr;
//                         if(cur_ptr >= vec_size){
//                            goto OUT;
//                         }else if(vec[cur_ptr]>max_start_id){
//                             max_start_id = vec[cur_ptr];
//                             goto RESCAN;
//                         }else{
//                             ptr[j]++;
//                         }
//                     }
//                     candidates_offset_map[next_depth].push_back(max_start_id);
//                     candidates[next_depth].push_back(aux_content[next_u].candidate[max_start_id]);
//                     if(ptr[0]<can[0].second){
//                         max_start_id = can[0].first[ptr[0]];
//                     }else{
//                         break;
//                     }
//                 }
//                 num_candidates = candidates[next_depth].size();
//                 if(enable_edge_order && next_depth != query_vertex_count){
//                     candidate_score.resize(num_candidates);
//                     memset(&candidate_score[0], 0, sizeof(float)*num_candidates);
//                     Vertex last_parent = *(pre[next_u].rbegin());
//                     vector<float>& score = aux_content[last_parent].neighbor_candidates[next_u][embedding_offset[last_parent]].second;
//                     for(uint32_t i=0; i<num_candidates; ++i){
//                         candidate_score[i] += score[embedding_offset[last_parent]];
//                     }
//                     // for(auto parent : pre[next_u]){
//                     //     vector<float>& score = aux_content[parent].neighbor_candidates[next_u][embedding_offset[parent]].second;
//                     //     for(uint32_t i=0; i<num_candidates; ++i){
//                     //         candidate_score[i] += score[embedding_offset[parent]];
//                     //     }
//                     // }
//                     // reorder the candidates
//                     for(int i=0;i<num_candidates-1;++i){
//                         for(int j=i+1;j<num_candidates;++j){
//                             if(candidate_score[i]<candidate_score[j]){
//                                 swap(candidate_score[i],candidate_score[j]);
//                                 swap(candidates_offset_map[next_depth][i], candidates_offset_map[next_depth][j]);
//                                 swap(candidates[next_depth][i], candidates[next_depth][j]);
//                             }
//                         }
//                     }
//                 }
//                 OUT:
//                 candidates_offset[next_depth] = 0;

// #ifdef FAILING_SET_PRUNING
//                 if(candidates[next_depth].size() == 0){
//                     // belongs to emptyset-class
//                     failing_set[next_depth] = ancestors[next_u];
//                     find_matches[next_depth] = false;
//                     // cout<<"empty_set:"<<next_depth<<endl;
//                 }
// #endif
//                 cur_depth ++;
//             }
//         }
        
// #ifdef FAILING_SET_PRUNING
//         // update the failing_set of the parent
//         if(find_matches[cur_depth] == true){
//             failing_set[cur_depth-1].reset();
//         }else if(find_matches[cur_depth-1] == false){
//             if(failing_set[cur_depth].test(u) == false){
//                 failing_set[cur_depth-1] = failing_set[cur_depth];
//                 // exit(0);
//             }else{
//                 failing_set[cur_depth-1] |= failing_set[cur_depth];
//             }
//         }
        
// #endif
//         cur_depth --;
//         if(cur_depth == 0){
//             break;
//         }
// #ifdef FAILING_SET_PRUNING
//         find_matches[cur_depth-1] |= find_matches[cur_depth];
//         // validate the failing set
//         if(find_matches[cur_depth]==false && failing_set[cur_depth].test(order[cur_depth]) == false){
//             // filtering the redundant siblings
//             candidates_offset[cur_depth] = candidates[cur_depth].size();
//             // cout<<"failing set pruning :"<<cur_depth<<":"<<emb_count<<endl;
//             failing_set[cur_depth-1] |= failing_set[cur_depth];
//         }
// #endif

//         visited_vertices[embedding[order[cur_depth]]] = false;
//     }
// EXIT:
//     delete [] candidates_offset;
//     delete [] embedding;
//     delete [] embedding_offset;
//     delete [] visited_vertices;
//     delete [] visited_query_vertices;
//     delete [] find_matches;
//     gettimeofday(&end_t, NULL);
//     if(stop == false){
//         timer_delete(id);
//     }
//     result_count = emb_count;
//     enumeration_time = get_time(start_t, end_t);
// }

