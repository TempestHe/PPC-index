#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <cstdlib>
#include <ctime>

#include "graph.hpp"
#include "embedding.hpp"
#include "feature_counter.hpp"

using namespace std;

class label_path{
public:
    vector<Label> content;

    label_path(vector<Label> c){
        copy(c.begin(), c.end(), inserter(content, content.begin()));
    }

    label_path(const label_path& l){
        content = l.content;
    }

    void push_back(const Label& v){
        content.push_back(v);
    }

    void reverse_content(){
        reverse(content.begin(), content.end());
    }

    bool operator==(const label_path& c) const{
        if(c.content.size() != content.size()){
            return false;
        }
        for(int i=0;i<content.size();++i){
            if(content[i] != c.content[i]){
                return false;
            }
        }
        return true;
    }
};

struct hash_label_path 
{
    size_t operator()(const label_path& c) const {
        if(c.content.size() == 0){
            return 0;
        }
        size_t h = hash<int>()(c.content[0]);
        for(int i=1; i<c.content.size(); ++i){
            h = h^hash<int>()(c.content[i]);
        }
        return h;
    }
};

bool reverse_path(vector<Label>& path, vector<Label>& reversed_path){
    reversed_path.clear();
    reversed_path.assign(path.begin(), path.end());
    reverse(reversed_path.begin(), reversed_path.end());
    bool is_same = true;
    for(int i=0;i<path.size();++i){
        if(path[i]!=reversed_path[i]){
            is_same = false;
            break;
        }
    }
    if(is_same == true){
        return false;
    }
    return true;
}



class Cycle_counter:public feature_counter{
public:
    bool enable_residual;
    int feature_type = 1; //cycle

    vector<vector<Label>> slot_labels;
    vector<vector<Label>> original_features;
    vector<vector<Label>> counting_features;
    vector<vector<Label>> extended_features;

    vector<unordered_map<Label, vector<Value>>> mask_maps;

    // utilized for edge
    vector<uint32_t> extended_slot_map; // if the j-th feature is a mirror feature, the value is the position of the original feature

    // utilized for vertex
    vector<Label> front_feature_vertex;

    // memory used for computations
    // Tensor** history_result=NULL;
    // Tensor* tensor_prev = NULL;
    // Tensor* tensor_prev_reverse = NULL;
    // Tensor* tensor_cur = NULL;
    // Tensor* tensor_cur_reverse = NULL;
    // int history_result_size = 0;

    
    Cycle_counter(bool enable_residual_, vector<vector<Label> >& original_features_){
        original_features = original_features_;
        enable_residual = enable_residual_;
        feature_initialization();
        construct_mask_map();
    }

    Cycle_counter(const Cycle_counter& other){
        original_features = other.original_features;
        enable_residual = other.enable_residual;
        feature_initialization();
        construct_mask_map();
    }

    Cycle_counter(feature_counter* other){
        original_features = other->get_features();
        enable_residual = other->get_residual();
        feature_initialization();
        construct_mask_map();
    }

    int get_feature_type(){
        return feature_type;
    }

    vector<vector<Label>> get_features(){
        return original_features;
    }

    bool get_residual(){
        return enable_residual;
    }

    // void pre_alloc(int edge_count, int vertex_count, int level){
    //     if(history_result != NULL){
    //         return;
    //     }
    //     if(enable_residual){
    //         history_result_size = original_features[0].size();
    //     }else{
    //         history_result_size = 1;
    //     }
    //     history_result = new Tensor* [history_result_size];
    //     tensor_prev = new Tensor(edge_count, extended_features.size(), 1);
    //     tensor_prev_reverse = new Tensor(edge_count, extended_features.size(), 1);
    //     tensor_cur = new Tensor(edge_count, extended_features.size(), 1);
    //     tensor_cur_reverse = new Tensor(edge_count, extended_features.size(), 1);
    //     for(int i=0;i<history_result_size; ++i){
    //         if(level == 1){
    //             history_result[i] = new Tensor(edge_count, original_features.size());
    //         }else{
    //             history_result[i] = new Tensor(vertex_count, original_features.size());
    //         }
    //     }
    // }

    // void release_alloc(){
    //     for(int i=0;i<history_result_size;++i){
    //         delete history_result[i];
    //     }
    //     delete [] history_result;
    //     history_result = NULL;
    //     delete tensor_cur;
    //     delete tensor_cur_reverse;
    //     delete tensor_prev;
    //     delete tensor_prev_reverse;
    // }


    void feature_initialization(){
        counting_features = original_features;
        extended_slot_map.resize(original_features.size()*2);
        extended_features.assign(original_features.begin(), original_features.end());
        // firstly extend the slot labels
        uint32_t count = 0;
        for(int i=0;i<original_features.size();++i){
            vector<Label> reversed_path = original_features[i];
            reverse(reversed_path.begin(), reversed_path.end());
            extended_features.push_back(reversed_path);
            // if(reverse_path(new_features[i], reversed_path)){
            //     extended_features.push_back(reversed_path);
            //     extended_slot_map[count+new_features.size()] = i;
            //     ++ count;
            // }
        }
        slot_labels.reserve(extended_features[0].size()-1);
        for(int l=extended_features[0].size()-1; l>=0; --l){
            vector<Label> inner;
            inner.reserve(extended_features.size());
            for(int j=0; j < extended_features.size(); ++j){
                inner.push_back(extended_features[j][l]);
            }
            slot_labels.push_back(inner);
        }
    }

    void construct_mask_map(){
        int feature_length = original_features[0].size();
        mask_maps.reserve(feature_length);
        for(int depth=0;depth<feature_length;++depth){
            int dim = slot_labels[depth].size();
            unordered_set<Label> labels;
            unordered_map<Label, vector<Value>> mask_map;
            for(int i=0; i<dim; ++i){
                Label label = slot_labels[depth][i];
                labels.insert(label);
            }
            for(auto label : labels){
                vector<Value> mask;
                for(int i=0; i<dim; ++i){
                    if(slot_labels[depth][i] == label){
                        mask.push_back(1);
                    }else{
                        mask.push_back(0);
                    }
                }
                mask_map.insert({label, mask});
            }
            mask_maps.push_back(mask_map);
        }
    }

    Tensor* merge_cycle_tensor_for_edges(Tensor* cycle_tensor, Tensor* reversed_cycle_tensor){
        // cout<<"history_result_size:"<<history_result_size<<":"<<counting_features.size()<<":"<<extended_features.size()<<endl;
        // Tensor* result = history_result[iteration];
        Tensor* result = new Tensor(cycle_tensor->row_size, original_features.size());
        int counting_features_num = counting_features.size();
        int extended_features_num = extended_features.size();
        Value** cycle_content = cycle_tensor->content;
        Value** cycle_reverse_content = reversed_cycle_tensor->content;
        for(int i=0; i<cycle_tensor->row_size; ++i){
            Value* r = result->content[i];
            vector_add(r, cycle_content[i], counting_features_num);
            vector_add(r, cycle_content[i]+counting_features_num, counting_features_num);
            vector_add(r, cycle_reverse_content[i], counting_features_num);
            vector_add(r, cycle_reverse_content[i]+counting_features_num, counting_features_num);
        }
        return result;
    }

    Tensor* merge_cycle_tensor_for_vertices(Graph& graph, Tensor* cycle_tensor, Tensor* reversed_cycle_tensor, unordered_map<Label, vector<Value>>& mask_map){
        int feature_size = original_features.size();
        Tensor* result = new Tensor(graph.adj.size(), feature_size);
        Value** result_content = result->content;
        Value** cycle_content = cycle_tensor->content;
        Value** reversed_cycle_content = reversed_cycle_tensor->content;
        for(Vertex v=0; v<graph.adj.size(); ++v){
            Value* result_vec = result_content[v];
            for(auto n : graph.adj[v]){
                auto itf = mask_map.find(graph.label_map[n]);
                if(itf == mask_map.end()){
                    continue;
                }
                vector<Value>& mask = itf->second;
                if(v<n){
                    Vertex e_id = graph.edge_id_map[v][n];
                    vector_add_mul(result_vec, reversed_cycle_content[e_id], &(mask[0]), feature_size);
                    vector_add_mul(result_vec, cycle_content[e_id]+feature_size, &(mask[feature_size]), feature_size);
                }else{
                    Vertex e_id = graph.edge_id_map[n][v];
                    vector_add_mul(result_vec, cycle_content[e_id], &(mask[0]), feature_size);
                    vector_add_mul(result_vec, reversed_cycle_content[e_id]+feature_size, &(mask[feature_size]), feature_size);
                }
            }
        }
        return result;
    }

    void count_for_vertices(Graph& graph, Tensor**& result, int& result_size){
#ifdef ENABLE_TIME_INFO
        struct timeval start_t, end_t;
#endif
        // start counting
        if(enable_residual){
            result_size = original_features[0].size();
        }else{
            result_size = 1;
        }
        result = new Tensor* [result_size];
        uint32_t edge_count = graph.get_edge_count();
        Tensor* tensor_prev= new Tensor(edge_count, extended_features.size(), 1); // v<-n
        Tensor* tensor_prev_reverse = new Tensor(edge_count, extended_features.size(), 1); // v->n
        Tensor* tensor_cur = new Tensor(edge_count, extended_features.size(), 0);
        Tensor* tensor_cur_reverse = new Tensor(edge_count, extended_features.size(), 0);
        int total_iterations = original_features[0].size();
        int offset = 0;
        if(enable_residual || total_iterations == 1){
            result[offset++] = merge_cycle_tensor_for_vertices(graph, tensor_prev, tensor_prev_reverse, mask_maps[0]);
        }
        for(int iteration=0; iteration<total_iterations-1; ++iteration){
#ifdef ENABLE_TIME_INFO
            cout<<"iteration:"<<iteration<<endl;
            gettimeofday(&start_t, NULL);
#endif
            // initialize the mask
            unordered_map<Label, vector<Value> >& mask_map = mask_maps[iteration];

            // iterate the edges in order
            Value** embeddings_prev = tensor_prev->content;
            Value** embeddings_prev_reverse = tensor_prev_reverse->content;
            for(Vertex e_id=0; e_id<edge_count; ++e_id){

                Value* embedding_cur = tensor_cur->content[e_id];
                Value* embedding_cur_reverse = tensor_cur_reverse->content[e_id];
                Vertex small_id = graph.edge_set[e_id*2];
                Vertex large_id = graph.edge_set[e_id*2+1];

                Vertex* n_neighbors = graph.common_edge_neighbor[e_id];
                Vertex neighbor_size = n_neighbors[0];
                for(Vertex itr=0; itr<neighbor_size; ++itr){
                    Vertex n = n_neighbors[itr*3+1];
                    Vertex small_e_id = n_neighbors[itr*3+2];
                    Vertex large_e_id = n_neighbors[itr*3+3];
                    auto itf = mask_map.find(graph.label_map[n]);
                    if(itf == mask_map.end()){
                        continue;
                    }
                    vector<Value>& mask = itf->second;
                    if(n<small_id){
                        vector_add_mul(embedding_cur_reverse, embeddings_prev[small_e_id], &(mask[0]), mask.size());
                    }else{
                        vector_add_mul(embedding_cur_reverse, embeddings_prev_reverse[small_e_id], &(mask[0]), mask.size());
                    }
                    
                    if(n<large_id){
                        vector_add_mul(embedding_cur, embeddings_prev[large_e_id], &(mask[0]), mask.size());
                    }else{
                        vector_add_mul(embedding_cur, embeddings_prev_reverse[large_e_id], &(mask[0]), mask.size());
                    }
                }
            }
            // swap the embedding
            tensor_prev->clear_content();
            tensor_prev_reverse->clear_content();
            if(enable_residual == true && iteration < total_iterations-2){
                Tensor* inner_result;
                result[offset++] = merge_cycle_tensor_for_vertices(graph, tensor_cur, tensor_cur_reverse, mask_maps[iteration+1]);
            }
            swap(tensor_prev, tensor_cur);
            swap(tensor_prev_reverse, tensor_cur_reverse);
#ifdef ENABLE_TIME_INFO
            gettimeofday(&end_t, NULL);
            cout<<"iteration:"<<iteration<<":"<<get_time(start_t, end_t)<<endl;
#endif
        }
        if(total_iterations > 1){
            result[offset++] = merge_cycle_tensor_for_vertices(graph, tensor_prev, tensor_prev_reverse, mask_maps[total_iterations-1]);
        }
        delete tensor_prev;
        delete tensor_prev_reverse;
        delete tensor_cur;
        delete tensor_cur_reverse;
    }



    void count_for_edges(Graph& graph, Tensor**& result, int& result_size){
#ifdef ENABLE_TIME_INFO
        struct timeval start_t, end_t;
#endif
        if(enable_residual){
            result_size = original_features[0].size();
        }else{
            result_size = 1;
        }
        result = new Tensor* [result_size];
        uint32_t edge_count = graph.get_edge_count();
        Tensor* tensor_prev= new Tensor(edge_count, extended_features.size(), 1); // v<-n
        Tensor* tensor_prev_reverse = new Tensor(edge_count, extended_features.size(), 1); // v->n
        Tensor* tensor_cur = new Tensor(edge_count, extended_features.size(), 0);
        Tensor* tensor_cur_reverse = new Tensor(edge_count, extended_features.size(), 0);
        int total_iterations = extended_features[0].size();
        // int edge_count = graph.get_edge_count();
        int offset = 0;
        for(int iteration=0; iteration<total_iterations; ++iteration){
#ifdef ENABLE_TIME_INFO
            cout<<"iteration:"<<iteration<<endl;
            gettimeofday(&start_t, NULL);
#endif
            // initialize the mask
            unordered_map<Label, vector<Value> >& mask_map = mask_maps[iteration];

            // iterate the edges in order
            Value** embeddings_prev = tensor_prev->content;
            Value** embeddings_prev_reverse = tensor_prev_reverse->content;
            for(Vertex e_id=0; e_id<edge_count; ++e_id){

                Value* embedding_cur = tensor_cur->content[e_id];
                Value* embedding_cur_reverse = tensor_cur_reverse->content[e_id];
                Vertex small_id = graph.edge_set[e_id*2];
                Vertex large_id = graph.edge_set[e_id*2+1];

                Vertex* n_neighbors = graph.common_edge_neighbor[e_id];
                Vertex neighbor_size = n_neighbors[0];
                for(Vertex itr=0; itr<neighbor_size; ++itr){
                    Vertex n = n_neighbors[itr*3+1];
                    Vertex small_e_id = n_neighbors[itr*3+2];
                    Vertex large_e_id = n_neighbors[itr*3+3];
                    auto itf = mask_map.find(graph.label_map[n]);
                    if(itf == mask_map.end()){
                        continue;
                    }
                    vector<Value>& mask = itf->second;
                    if(n<small_id){
                        vector_add_mul(embedding_cur_reverse, embeddings_prev[small_e_id], &(mask[0]), mask.size());
                    }else{
                        vector_add_mul(embedding_cur_reverse, embeddings_prev_reverse[small_e_id], &(mask[0]), mask.size());
                    }
                    
                    if(n<large_id){
                        vector_add_mul(embedding_cur, embeddings_prev[large_e_id], &(mask[0]), mask.size());
                    }else{
                        vector_add_mul(embedding_cur, embeddings_prev_reverse[large_e_id], &(mask[0]), mask.size());
                    }
                }
            }
            // swap the embedding
            tensor_prev->clear_content();
            tensor_prev_reverse->clear_content();
            if(enable_residual == true && iteration < total_iterations-1){
                Tensor* inner_result;
                inner_result = merge_cycle_tensor_for_edges(tensor_cur, tensor_cur_reverse);
                result[offset++] = inner_result;
                // merge_cycle_tensor_for_edges(tensor_cur, tensor_cur_reverse, iteration);
            }
            swap(tensor_prev, tensor_cur);
            swap(tensor_prev_reverse, tensor_cur_reverse);
#ifdef ENABLE_TIME_INFO
            gettimeofday(&end_t, NULL);
            cout<<"iteration:"<<iteration<<":"<<get_time(start_t, end_t)<<endl;
#endif
        }
        // merge_cycle_tensor_for_edges(tensor_prev, tensor_prev_reverse, total_iterations-1);
        result[offset++] = merge_cycle_tensor_for_edges(tensor_prev, tensor_prev_reverse);
        delete tensor_prev;
        delete tensor_prev_reverse;
        delete tensor_cur;
        delete tensor_cur_reverse;
    }
};
