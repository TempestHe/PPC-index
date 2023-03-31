import argparse 
import os
import json
import ipdb
import torch
import struct
import time
import random


import networkx as nx

import sys
sys.path.append("../../index")
from ppc_index import *
sys.path.append("../")
from feature_extractor import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help="query")
    parser.add_argument('--input_data', type=str, help="data")
    # [] means vanilla vc-framework
    parser.add_argument('--para', type=str, default="['64-4-1-1-1']") # feature_num feature_length residual feature_type(0 path, 1 cycle) level(0 vertex 1 edges)
    args = parser.parse_args()
    return args


def parse_graph(filename):
    graph_list = []
    with open(filename) as f:
        G = None
        for line in f.readlines():
            if line[0] == '#':
                if G is not None:
                    graph_list.append(G)
                G = nx.Graph()
            elif line[0] == 'v':
                ele = line.strip("\n").split(" ")
                G.add_node(int(ele[1]), label=int(ele[2]))
            elif line[0] == 'e':
                ele = line.strip("\n").split(" ")
                G.add_edge(int(ele[1]), int(ele[2]))
        if len(G.nodes()) > 0:
            graph_list.append(G)
    return graph_list

def parse_samples(filename):
    graph_list = []
    anchors = [[], []]
    with open(filename) as f:
        G = None
        for line in f.readlines():
            if line[0] == '#':
                if G is not None:
                    graph_list.append((G, [anchors[0][:], anchors[1][:]]))
                G = nx.Graph()
                anchors[0].clear()
                anchors[1].clear()
            elif line[0] == 'v':
                ele = line.strip("\n").split(" ")
                G.add_node(int(ele[1]), label=int(ele[2]))
            elif line[0] == 'e':
                ele = line.strip("\n").split(" ")
                G.add_edge(int(ele[1]), int(ele[2]))
            elif line[0] == 'q':
                ele = line.strip("\n").split(" ")
                anchors[0].append(int(ele[1]))
            elif line[0] == 'd':
                ele = line.strip("\n").split(" ")
                anchors[1].append((int(ele[1]), int(ele[2])))
        if len(G.nodes()) > 0:
            graph_list.append((G, [anchors[0][:], anchors[1][:]]))
    return graph_list

def dump_file(filename, graph_pair_list):
    with open(filename, 'w') as f:
        for id, graph_pair in enumerate(graph_pair_list):
            graph = graph_pair[0]
            f.write("# {0}\n".format(id))
            for n in graph.nodes():
                f.write("v {0} {1}\n".format(n, graph.nodes[n]["label"]))
            for e in graph.edges():
                f.write("e {0} {1}\n".format(e[0], e[1]))
            for a_q, a_d in zip(graph_pair[1][0], graph_pair[1][1]):
                f.write("q {0}\n".format(a_q))
                f.write("d {0} {1}\n".format(a_d[0], a_d[1]))

def parse_parameters(paras, index_path, parsed_args):
    index_files = []
    counter_paras = []
    for para in paras:
        feature_type = 'path' if para[3]==0 else 'cycle'
        level = "edge" if para[4] == 1 else "vertex"

        name = parsed_args.input_data.split("/")[-1]+"_"+str(para[0])+"_"+str(para[1])+"_"+str(para[2])+"_"+feature_type+"_"+level
        level = "edge" if para[4] == 1 else "vertex"
        index_name = os.path.join(index_path, name+".index")
        index_features = os.path.join(index_path, name+".features")
        index_files.append(index_name)
        counter_paras.append(para+[index_features])
        if not os.path.exists(index_name):
            print("index file:"+index_name+" does not exists!")
            return None
            
    return index_files, counter_paras

def run_exp(parsed_args, index_path):
    index_file_list = []
    feature_file_list = []

    lines = eval(parsed_args.para)
    edge_paras = []
    vertex_paras = []
    for line in lines:
        p = line.split('-')
        if int(p[4]) == 0:
            vertex_paras.append([int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])])
        else:
            edge_paras.append([int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])])

    if len(vertex_paras)>0 and len(edge_paras)>0:
        print("both vertex and edge indicies are used, choose only one of them")
        exit(0)

    level = ""
    if len(vertex_paras) > 0:
        level = "vertex"
    else:
        level = "edge"

    edge_index_files = []
    vertex_index_file = []
    edge_counter_paras = []
    vertex_counter_paras = []
    # ipdb.set_trace()
    edge_index_files, edge_counter_paras = parse_parameters(edge_paras, index_path, parsed_args)
    vertex_index_file, vertex_counter_paras = parse_parameters(vertex_paras, index_path, parsed_args)

    index_parser_list = []
    counter_list = []
    if level == "vertex":
        for file in vertex_index_file:
            index_parser_list.append(IndexParser(file))
        for para in vertex_counter_paras:
            feature_file = para[-1]
            with open(feature_file) as f:
                features = json.load(f)
            if para[3] == 1:
                feature_type = 'cycle'
            else:
                feature_type = 'path'
            counter_list.append(FeatureCounter(para[2]==1, features, feature_type))
    else:
        for file in edge_index_files:
            index_parser_list.append(IndexParser(file))
        for para in edge_counter_paras:
            feature_file = para[-1]
            with open(feature_file) as f:
                features = json.load(f)
            if para[3] == 1:
                feature_type = 'cycle'
            else:
                feature_type = 'path'
            counter_list.append(FeatureCounter(para[2]==1, features, feature_type))

    data_graph = parse_graph(parsed_args.input_data)[0]
    if level == 'edge':
        data_edge_id_map = generate_edge_id_map(data_graph)


    results = {}
    for file in os.listdir(parsed_args.queries):
        if file.endswith(".json"):
            continue
        query_samples = parse_samples(os.path.join(parsed_args.queries, file))

        filtered_sample = []
        samples_for_selection = []

        if level == 'vertex' and len(query_samples[0][1][0]) > 1:
            continue
        elif level == 'edge' and len(query_samples[0][1][0]) == 1:
            continue

        results[file] = []
        filtered_samples = 0
        for sample in query_samples:
            query_embs = []
            data_embs = []
            query_graph = sample[0]
            query_anchor = sample[1][0][0]
            data_anchor = sample[1][1][0][0]
            
            if level == 'edge':
                query_anchor_u = sample[1][0][1]
                data_anchor_v = sample[1][1][1][0]

                query_edge_id_map = generate_edge_id_map(query_graph)
                data_e_id = data_edge_id_map[data_anchor][data_anchor_v] if data_anchor<data_anchor_v else data_edge_id_map[data_anchor_v][data_anchor]
                query_e_id = query_edge_id_map[query_anchor][query_anchor_u] if query_anchor<query_anchor_u else query_edge_id_map[query_anchor_u][query_anchor]
            
            for counter, index_parser in zip(counter_list, index_parser_list):
                emb = counter.count(query_graph, level, 1)
                if level == 'edge':
                    query_embs.append(emb[query_e_id])
                    data_embs.append(index_parser.load_vertex_embeddings(0, data_e_id))
                else:
                    query_embs.append(emb[query_anchor])
                    data_embs.append(index_parser.load_vertex_embeddings(0, data_anchor))
            
            # comparing the embeddings
            contained = True
            for q_emb, d_emb in zip(query_embs, data_embs):
                if q_emb.is_contained(d_emb)[0] == False:
                    contained = False
                    break

            if contained == False:
                filtered_samples += 1
                if len(filtered_sample) < 40:
                    filtered_sample.append(sample)
                else:
                    samples_for_selection.append(sample)
            else:
                samples_for_selection.append(sample)
        if len(filtered_sample) < 40:
            filtered_sample += samples_for_selection[:40-len(filtered_sample)]
            samples_for_selection = samples_for_selection[40-len(filtered_sample):]
        # dump_file(file+".selection", samples_for_selection)
        # dump_file(file+".refined", filtered_sample)

        results[file] = {"filtered_num":filtered_samples, "total_samples": len(query_samples)}
        print("{0} precision:{1}/{2}={3}".format(file, filtered_samples, len(query_samples), filtered_samples/float(len(query_samples))))

    filtered_sample_num_global = 0
    total_sample_num_global = 0
    for file in results:
        filtered_sample_num_global += results[file]["filtered_num"]
        total_sample_num_global += results[file]["total_samples"]
    print("average precision:{0}/{1}={2}".format(filtered_sample_num_global, total_sample_num_global, filtered_sample_num_global/float(total_sample_num_global)))
    results["average_precision"] = filtered_sample_num_global/float(total_sample_num_global)
    
    if level == 'vertex':
        vertex_para_name = []
        for para in vertex_paras:
            vertex_para_name.append("-".join([str(p) for p in para]))
        result_file_name = parsed_args.input_data.split("/")[-1]+"_"+"_".join(vertex_para_name)+".json"
    else:
        edge_para_name = []
        for para in edge_paras:
            edge_para_name.append("-".join([str(p) for p in para]))
        result_file_name = parsed_args.input_data.split("/")[-1]+"_"+"_".join(edge_para_name)+".json"
    
    if not os.path.exists(".result"):
        os.mkdir(".result")

    with open(os.path.join(".result", result_file_name), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    args = parse_args()
    default_index_path = ".index"

    run_exp(args, default_index_path)