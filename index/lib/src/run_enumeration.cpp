#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <dirent.h>
#include "subgraph_enumeration.h"
#include "subgraph_retrieval.h"
#include "../utility/utils.h"
#include "../utility/core_decomposition.h"

void load_graph_list(string filename, vector<Graph>& result, vector<vector<Vertex>>& query_anchors, vector<vector<Vertex>>& data_anchors){
    ifstream in(filename);

    string line;
    unordered_map<Vertex, Label> label_map_unordered;
    unordered_map<Vertex, unordered_set<Vertex>> adj_unordered;
    vector<Label> label_map;
    vector<unordered_set<Vertex>> adj;
    vector<Vertex> q_anchors;
    vector<Vertex> d_anchors;
    while(getline(in, line)){
        if(line.c_str()[0] == '#'){
            if(label_map_unordered.size()>0){
                label_map.resize(label_map_unordered.size());
                for(auto v : label_map_unordered){
                    label_map[v.first] = v.second;
                }
                adj.resize(label_map_unordered.size());
                for(auto v : adj_unordered){
                    adj[v.first] = v.second;
                }
                Graph graph(adj, label_map);
                result.push_back(graph);
                data_anchors.push_back(d_anchors);
                query_anchors.push_back(q_anchors);
                d_anchors.clear();
                q_anchors.clear();
                label_map.clear();
                adj.clear();
            }
            label_map_unordered.clear();
            adj_unordered.clear();
        }else if(line.c_str()[0] == 'v'){
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
        }else if(line.c_str()[0] == 'q'){
            stringstream ss(line);
            Vertex u;
            char c;
            ss>>c>>u;
            q_anchors.push_back(u);
        }else if(line.c_str()[0] == 'd'){
            stringstream ss(line);
            Vertex v;
            char c;
            ss>>c>>v;
            d_anchors.push_back(v);
        }
    }
    if(label_map_unordered.size()>0){
        label_map.resize(label_map_unordered.size());
        for(auto v : label_map_unordered){
            label_map[v.first] = v.second;
        }
        adj.resize(label_map_unordered.size());
        for(auto v : adj_unordered){
            adj[v.first] = v.second;
        }
        Graph graph(adj, label_map);
        result.push_back(graph);
        data_anchors.push_back(d_anchors);
        query_anchors.push_back(q_anchors);
    }
}

struct Param{
    string query_path;
    string data_file;
    string PV_data_index;
    string PV_feature;
    string PE_data_index;
    string PE_feature;
    string CV_data_index;
    string CV_feature;
    string CE_data_index;
    string CE_feature;
    bool enable_ordering;
    bool enable_residual;
    int count_limit;
};

static struct Param parsed_input_para;

static const struct option long_options[] = {
    {"query", required_argument, NULL, 'q'},
    {"data", required_argument, NULL, 'd'},
    {"PV", required_argument, NULL, 'x'},
    {"PE", required_argument, NULL, 'y'},
    {"CV", required_argument, NULL, 'm'},
    {"CE", required_argument, NULL, 'n'},
    {"order", required_argument, NULL, 'o'},
    {"residual", required_argument, NULL, 'r'},
    {"limit", required_argument, NULL, 'l'},
    {"help", no_argument, NULL, '?'},
};

void print_args(){
    cout<<"query:\t"<<parsed_input_para.query_path<<endl;
    cout<<"data:\t"<<parsed_input_para.data_file<<endl;
    cout<<"ordering:\t"<<parsed_input_para.enable_ordering<<endl;
    cout<<"residual:\t"<<parsed_input_para.enable_residual<<endl;
    cout<<"limit:\t"<<parsed_input_para.count_limit<<endl;
    cout<<"PV:\t"<<parsed_input_para.PV_data_index<<"\t"<<parsed_input_para.PV_feature<<endl;
    cout<<"PE:\t"<<parsed_input_para.PE_data_index<<"\t"<<parsed_input_para.PE_feature<<endl;
    cout<<"CV:\t"<<parsed_input_para.CV_data_index<<"\t"<<parsed_input_para.CV_feature<<endl;
    cout<<"CE:\t"<<parsed_input_para.CE_data_index<<"\t"<<parsed_input_para.CE_feature<<endl;
}

vector<string> GetFiles(string path, string suffix){
    if(path[path.size()-1] == '/'){
        path = path.substr(0, path.size()-1);
    }

    const char* src_dir = path.c_str();
    const char* ext = suffix.c_str();
    vector<string> result;
    string directory(src_dir);
    string m_ext(ext);

    DIR *dir = opendir(src_dir);
    if ( dir == NULL )
    {
        printf("[ERROR] %s is not a directory or not exist!", src_dir);
        return result;
    }
 
    struct dirent* d_ent = NULL;
 
    char dot[3] = ".";
    char dotdot[6] = "..";
 
    while ( (d_ent = readdir(dir)) != NULL )
    {
        if ( (strcmp(d_ent->d_name, dot) != 0) && (strcmp(d_ent->d_name, dotdot) != 0) )
        {
            if ( d_ent->d_type != DT_DIR)
            {
                string d_name(d_ent->d_name);
                if (strcmp(d_name.c_str () + d_name.length () - m_ext.length(), m_ext.c_str ()) == 0)
                {
                    result.push_back(path+string("/")+string(d_ent->d_name));
                }
            }
        }
    }
 
    // sort the returned files
    sort(result.begin(), result.end());
 
    closedir(dir);
    return result;
}

void parse_args(int argc, char** argv){
    int opt;
    int options_index=0;
    string suffix;
    parsed_input_para.enable_residual = 1;
    parsed_input_para.count_limit = 100000;
    while((opt=getopt_long_only(argc, argv, "q:d:x:y:m:n:?", long_options, &options_index)) != -1){
        switch (opt)
        {
        case 0:
            break;
        case 'q':
            parsed_input_para.query_path = string(optarg);
            break;
        case 'd':
            parsed_input_para.data_file = string(optarg);
            break;
        case 'x':
            parsed_input_para.PV_data_index = string(optarg);
            suffix = parsed_input_para.PV_data_index.substr(parsed_input_para.PV_data_index.size()-6, parsed_input_para.PV_data_index.size());
            if(suffix.compare(string(".index")) == 0){
                parsed_input_para.PV_feature = parsed_input_para.PV_data_index.substr(0, parsed_input_para.PV_data_index.size()-6)+string(".features");
            }else{
                cout<<"the suffix of the index file has to be '.index'"<<endl;
                exit(0);
            }
            break;
        case 'y':
            parsed_input_para.PE_data_index = string(optarg);
            suffix = parsed_input_para.PE_data_index.substr(parsed_input_para.PE_data_index.size()-6, parsed_input_para.PE_data_index.size());
            if(suffix.compare(string(".index")) == 0){
                parsed_input_para.PE_feature = parsed_input_para.PE_data_index.substr(0, parsed_input_para.PE_data_index.size()-6)+string(".features");
            }else{
                cout<<"the suffix of the index file has to be '.index'"<<endl;
                exit(0);
            }
            break;
        case 'm':
            parsed_input_para.CV_data_index = string(optarg);
            suffix = parsed_input_para.CV_data_index.substr(parsed_input_para.CV_data_index.size()-6, parsed_input_para.CV_data_index.size());
            if(suffix.compare(string(".index")) == 0){
                parsed_input_para.CV_feature = parsed_input_para.CV_data_index.substr(0, parsed_input_para.CV_data_index.size()-6)+string(".features");
            }else{
                cout<<"the suffix of the index file has to be '.index'"<<endl;
                exit(0);
            }
            break;
        case 'n':
            parsed_input_para.CE_data_index = string(optarg);
            suffix = parsed_input_para.CE_data_index.substr(parsed_input_para.CE_data_index.size()-6, parsed_input_para.CE_data_index.size());
            if(suffix.compare(string(".index")) == 0){
                parsed_input_para.CE_feature = parsed_input_para.CE_data_index.substr(0, parsed_input_para.CE_data_index.size()-6)+string(".features");
            }else{
                cout<<"the suffix of the index file has to be '.index'"<<endl;
                exit(0);
            }
            break;
        case 'l':
            parsed_input_para.count_limit = atoi(optarg);
            break;
        case 'o':
            parsed_input_para.enable_ordering = atoi(optarg);
            break;
        case 'r':
            parsed_input_para.enable_residual = atoi(optarg);
            break;
        case '?':
            cout<<"------------------ args list ------------------------"<<endl;
            cout<<"--query\tpath of the query graph"<<endl;
            cout<<"--data\tpath of the data graph"<<endl;
            cout<<"--order\t0/1 whether enable the candidate ordering mechanism by PPC-indices"<<endl;
            cout<<"--residual\t0/1 whether enable the residual mechanism during counting(default 1)"<<endl;
            cout<<"--limit\tnumber of found matches after return"<<endl;
            cout<<"--PV\t path of the vertex-leveled path index for data graph(optional)"<<endl;
            cout<<"--PE\t path of the edge-leveled path index for data graph(optional)"<<endl;
            cout<<"--CV\t path of the vertex-leveled cycle index for data graph(optional)"<<endl;
            cout<<"--CE\t path of the edge-leveled cycle index for data graph(optional)"<<endl;
            break;
        default:
            break;
        }
    }
}

Tensor* load_index(vector<Index_manager>& managers){
    Tensor* result = NULL;
    vector<Tensor*> vec;
    for(auto m : managers){
        vec.push_back(m.load_graph_tensor(0));
    }
    if(!vec.empty()){
        if(vec.size() > 1){
            result = merge_multi_Tensors(vec);
            for(auto e : vec){
                delete e;
            }
        }else{
            result = vec[0];
        }
    }
    return result;
}

Tensor* count_for_query(vector<Index_constructer>& con, Graph& query, int level){
    Tensor* result = NULL;
    vector<Tensor*> vec;
    for(auto m : con){
        vec.push_back(m.count_features(query, 1, level));
    }
    if(!vec.empty()){
        if(vec.size() > 1){
            result = merge_multi_Tensors(vec);
            for(auto e : vec){
                delete e;
            }
        }else{
            result = vec[0];
        }
    }
    return result;
}

int main(int argc, char** argv){
    parse_args(argc, argv);
    print_args();
    // initialize parameters
    vector<Index_manager> vertex_index;
    vector<Index_manager> edge_index;
    vector<Index_constructer> vertex_constructor;
    vector<Index_constructer> edge_constructor;

    if(!parsed_input_para.PV_data_index.empty()){
        vertex_index.emplace_back(Index_manager(parsed_input_para.PV_data_index));
        vector<vector<Label>> features = load_label_path(parsed_input_para.PV_feature);
        Path_counter* counter = new Path_counter(parsed_input_para.enable_residual, features);
        vertex_constructor.emplace_back(counter);
    }
    if(!parsed_input_para.CV_data_index.empty()){
        vertex_index.emplace_back(Index_manager(parsed_input_para.CV_data_index));
        vector<vector<Label>> features = load_label_path(parsed_input_para.CV_feature);
        Cycle_counter* counter = new Cycle_counter(parsed_input_para.enable_residual, features);
        vertex_constructor.emplace_back(counter);
    }
    if(!parsed_input_para.PE_data_index.empty()){
        edge_index.emplace_back(Index_manager(parsed_input_para.PE_data_index));
        vector<vector<Label>> features = load_label_path(parsed_input_para.PE_feature);
        Path_counter* counter = new Path_counter(parsed_input_para.enable_residual, features);
        vertex_constructor.emplace_back(counter);
    }
    if(!parsed_input_para.CE_data_index.empty()){
        edge_index.emplace_back(Index_manager(parsed_input_para.CE_data_index));
        vector<vector<Label>> features = load_label_path(parsed_input_para.CE_feature);
        Path_counter* counter = new Path_counter(parsed_input_para.enable_residual, features);
        vertex_constructor.emplace_back(counter);
    }
    
    // loading the indexes
    cout<<"start loading index"<<endl;
    Graph data_graph(parsed_input_para.data_file);
    Tensor* vertex_emb_d = load_index(vertex_index);
    Tensor* edge_emb_d = load_index(edge_index);
    cout<<"done loading index"<<endl;

    // start enumeration
    vector<string> query_files = GetFiles(parsed_input_para.query_path, string("gr"));
    double total_enumeration_time = 0;
    int total_queries = 0;
    for(auto query_file : query_files){
        vector<Graph> query_graphs;
        vector<vector<Vertex>> query_anchors, data_anchors; // useless
        load_graph_list(query_file, query_graphs, query_anchors, data_anchors);

        struct timeval start_t, end_t;
        int id = 0;
        for(auto query_graph : query_graphs){
            gettimeofday(&start_t, NULL);
            Tensor* vertex_emb_q = count_for_query(vertex_constructor, query_graph, 0);
            Tensor* edge_emb_q = count_for_query(edge_constructor, query_graph, 1);
            gettimeofday(&end_t, NULL);

            // start enumeration
            long result_count;
            double enumeration_time, preprocessing_time;
            // cout<<vertex_emb_d->to_string()<<endl;
            subgraph_enumeration(&data_graph, &query_graph, 
                parsed_input_para.count_limit, result_count, 
                enumeration_time, preprocessing_time, 
                vertex_emb_q, vertex_emb_d,
                edge_emb_q, edge_emb_d,
                parsed_input_para.enable_ordering // whether order the candidate
            );
            cout<<query_file<<"-"<<id<<":"<<"query_indexing_time:"<<get_time(start_t, end_t)<<" result:"<<result_count<<" preprocessing_time:"<<preprocessing_time<<" enumeration_time:"<<enumeration_time<<endl;
            delete vertex_emb_q, edge_emb_q;
            total_enumeration_time += enumeration_time;
            total_queries += 1;
            id++;
        }
    }
    cout<<"average enumeration time:"<<total_enumeration_time/total_queries<<endl;
}
