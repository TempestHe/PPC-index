#pragma once
#include "auxiliary.h"
#include "../index/feature_counter.h"
#include "../index/cycle_counting.h"
#include "../index/path_counting.h"
#include "../index/index.h"
#include "../utility/embedding.h"
#include "../utility/utils.h"
#include "../graph/graph.h"

pair<uint32_t*, int> sum_tensor_safe_without_overflow(Tensor* tensor);

bool validate_tensor_safe_without_overflow(uint32_t* t1, uint32_t* t2, int size);

vector<Tensor*> load_multi_index(vector<string>& file_list, vector<Graph>& graphs);

// only support maximally two gnns with two different types
void sub_containment_vc(vector<Graph>& data_graphs, vector<Graph>& query_graphs, vector<int>& filtered_results, vector<int>& ground_truth, vector<double>& filtering_time);

void sub_containment_vc_index(vector<Graph>& data_graphs, vector<Graph>& query_graphs, 
    vector<string>& index_edge_list, vector<string>& index_vertex_list, 
    vector<feature_counter*>& counter_edge_list, vector<feature_counter*>& counter_vertex_list,
    vector<int>& filtered_results, vector<int>& ground_truth, vector<double>& filtering_time
);

void sub_containment_graph_level(vector<Graph>& data_graphs, vector<Graph>& query_graphs, 
    vector<string>& index_edge_list, vector<string>& index_vertex_list, 
    vector<feature_counter*>& counter_edge_list, vector<feature_counter*>& counter_vertex_list,
    vector<int>& filtered_results, vector<int>& ground_truth, vector<double>& filtering_time
);
