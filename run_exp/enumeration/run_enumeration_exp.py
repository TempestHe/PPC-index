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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help="query")
    parser.add_argument('--input_data', type=str, help="data")
    parser.add_argument('--para', type=str, default="['64-4-1-1-1']") # feature_num feature_length residual feature_type(0 path, 1 cycle) level(0 vertex 1 edges) # if para=[], means no index is involved
    parser.add_argument('--enable_order', type=bool, help="data") # whether enable candidate ordering
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

def run_exp(parsed_args, index_path, result_path):
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

    edge_index_files = []
    vertex_index_file = []
    edge_counter_paras = []
    vertex_counter_paras = []

    edge_index_files, edge_counter_paras = parse_parameters(edge_paras, index_path, parsed_args)
    vertex_index_file, vertex_counter_paras = parse_parameters(vertex_paras, index_path, parsed_args)
    
    data_graph = parse_graph(parsed_args.input_data)[0]

    result = {}
    for file in os.listdir(parsed_args.queries):
        if file.endswith(".json"):
            continue
        query_graphs = [s[0] for s in parse_samples(os.path.join(parsed_args.queries, file))]

        result[file] = run_subgraph_enumeration(file, data_graph, query_graphs, parsed_args.enable_order, 100000, edge_index_files, vertex_index_file, edge_counter_paras, vertex_counter_paras)

    edge_para_name = []
    for para in edge_paras+vertex_paras:
        edge_para_name.append("-".join([str(p) for p in para]))
    order = "order_enable" if parsed_args.enable_order else "order_disable"
    result_file_name = parsed_args.input_data.split("/")[-1]+"_"+"_".join(edge_para_name)+"_"+order+".json"
    
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(os.path.join(result_path, result_file_name), 'w') as f:
        json.dump(result, f)

    

if __name__ == '__main__':
    args = parse_args()
    print("loading queries")

    default_index_path = ".index"
    default_result_path = ".index_enum"
    if not os.path.exists(default_index_path):
        os.mkdir(default_index_path)
    run_exp(args, default_index_path, default_result_path)