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
    parser.add_argument('--sample', type=str, help="query")
    parser.add_argument('--input_data', type=str, help="data")
    parser.add_argument('--para', type=str, default="64-4-1-1-1") # feature_num feature_length residual feature_type(0 path, 1 cycle) level(0 vertex 1 edges)
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
    print(filename)
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

def build_index(parsed_args, index_path):
    index_file_list = []
    feature_file_list = []

    p = parsed_args.para.split('-')
    paras = [int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])]

    # get the max_label
    data_graphs = parse_graph(parsed_args.input_data)
    max_label = 0
    for graph in data_graphs:
        for v in graph.nodes():
            if graph.nodes[v]['label'] > max_label:
                max_label = graph.nodes[v]['label']

    feature_type = 'path' if paras[3]==0 else 'cycle'
    level = "edge" if paras[4] == 1 else "vertex"
    
    name = parsed_args.input_data.split("/")[-1]+"_"+str(paras[0])+"_"+str(paras[1])+"_"+str(paras[2])+"_"+feature_type+"_"+level
    level = "edge" if paras[4] == 1 else "vertex"
    index_name = os.path.join(index_path, name+".index")
    feature_name = os.path.join(index_path, name+".features")
    if not os.path.exists(feature_name):
        samples = []
        for file in os.listdir(parsed_args.sample):
            filename = os.path.join(parsed_args.sample, file)
            s = parse_samples(filename)
            if len(s[0][1][0]) == 2 and level == 'vertex':
                print("sample:"+filename+" is not a vertex level sample")
                exit(0)
            elif len(s[0][1][0]) == 1 and level == 'edge':
                print("sample:"+filename+" is not a edge level sample")
                exit(0)
            samples += s
        
        feature_extractor = FeatureExtractor(max_label+1, paras[1], paras[0], (paras[2]==1), (paras[3]==1), level, 60)
        features = feature_extractor.extract_for_singe_graph_large(samples, data_graphs[0], level, thread_num=12)
        
        feature_extractor.dump_features(features, feature_name)
    else:
        with open(feature_name) as f:
            features = json.load(f)

    if os.path.exists(index_name):
        print("index file:"+index_name+" already exists!")
        return
    
    
    counter = FeatureCounter((paras[2]==1), features, feature_type)
    build_time = counter.build_index(data_graphs, index_name, level, 1, 128)
    with open(index_name+".json", 'w') as f:
        json.dump({"build time": build_time}, f)

if __name__ == '__main__':
    args = parse_args()
    print("loading queries")

    default_index_path = ".index"
    if not os.path.exists(default_index_path):
        os.mkdir(default_index_path)
    
    build_index(args, default_index_path)