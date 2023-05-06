#pragma once
#include "auxiliary.h"
#include "../index/feature_counter.h"
#include "../index/cycle_counting.h"
#include "../index/path_counting.h"
#include "../utility/embedding.h"
#include "../utility/utils.h"
#include "../graph/graph.h"




void set_stop_flag_enum(union sigval val);
void register_timer_enum(bool* stop_flag);

void subgraph_enumeration_get_candidate_info(Graph* data_graph, Graph* query_graph, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    uint32_t& vertex_candidate_num, uint32_t& edge_candidate_num
);

bool subgraph_containment_test(Graph* data_graph, Graph* query_graph);

void subgraph_enumeration(Graph* data_graph, Graph* query_graph, 
    long count_limit, long& result_count, 
    double& enumeration_time, double& preprocessing_time, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    bool enable_order // whether order the candidates
);