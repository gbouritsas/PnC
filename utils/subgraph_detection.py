import torch
import numpy as np

import graph_tool as gt
import graph_tool.stats as gt_stats
import graph_tool.topology as gt_topology

from tqdm import tqdm
import datetime

import xlrd
from xlutils.copy import copy
from xlwt import Workbook

import os
import pickle
import utils.parsing as parse
from joblib import delayed, Parallel
import time

def subgraph_isomorphism(graph, H, directed=False, induced=True):
    
    
    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(graph.edge_index.transpose(1,0).cpu().numpy()))
    gt_stats.remove_self_loops(G_gt)
    gt_stats.remove_parallel_edges(G_gt)  
       
        
    sub_iso = gt_topology.subgraph_isomorphism(H, G_gt, induced=induced, subgraph=True, generator=True) # compute all subgraph isomorphisms

    V_S_set = set() # auxilliary: set of subgraph vertex sets V_S. Used to avoid repetitions due to subgraph automorphisms
    V_S_mappings = [] # m x |V_H|: matched subgraphs with label mapping
    for sub_iso_curr in sub_iso:
        sub_iso_curr = np.array(sub_iso_curr.get_array())
        if frozenset(sub_iso_curr) not in V_S_set:
            V_S_set.add(frozenset(sub_iso_curr))
            V_S_mappings.append(sub_iso_curr)
    V_S_mappings = np.array(V_S_mappings)

    return V_S_mappings, V_S_set


def detect_load_subgraphs(graphs_ptg, data_folder, H_set, directed, multiprocessing, num_processes):
    if directed:
        subgraph_files_list = os.path.join(data_folder, 'directed_subgraph_files_list.xls')
    else:
        subgraph_files_list = os.path.join(data_folder, 'subgraph_files_list.xls')
    if not os.path.exists(subgraph_files_list):
        wb = Workbook()
        w_sheet = wb.add_sheet('Sheet 1')
        wb.save(subgraph_files_list)
        saved_edge_lists = []
        sheet_rows = 0
    else:
        rb = xlrd.open_workbook(subgraph_files_list)
        wb = copy(rb)
        r_sheet = rb.sheet_by_index(0)
        w_sheet = wb.get_sheet(0)
        saved_edge_lists = [r_sheet.cell_value(i, 0) for i in range(r_sheet.nrows)]
        sheet_rows = r_sheet.nrows
    V_S_mappings_all = []
    for i, H in enumerate(H_set):
        edge_list = H.get_edges()
        edge_list_str = parse.ListOfListsOfint2str(edge_list)
        if edge_list_str in saved_edge_lists:
            print("Loading subgraphs...")
            subgraph_file = r_sheet.cell_value(saved_edge_lists.index(edge_list_str), 1)
            with open(subgraph_file, 'rb') as f:
                V_S_mappings_H, V_S_set_H = pickle.load(f)
            V_S_mappings_all.append(V_S_mappings_H)

        else:
            V_S_mappings_H, V_S_set_H = [], []
            ### parallel computation of subgraph isomorphisms
            if multiprocessing:
                print("Detecting subgraphs in parallel...")
                start = time.time()
                V_S = Parallel(n_jobs=num_processes, verbose=10)(delayed(subgraph_isomorphism)(graph,
                                                                                               H_set[i],
                                                                                               directed=directed) for
                                                                 graph in graphs_ptg)
                V_S_mappings_H = [V_S_i[0] for V_S_i in V_S]
                V_S_set_H = [V_S_i[1] for V_S_i in V_S]
                print('Done ({:.2f} secs).'.format(time.time() - start))
            ### single-threaded computation of subgraph isomoprhisms
            else:
                print("Detecting subgraphs...")
                for graph in tqdm(graphs_ptg):
                    V_S_mappings_i, V_S_set_i = subgraph_isomorphism(graph, H_set[i], directed=directed)
                    V_S_mappings_H.append(V_S_mappings_i)
                    V_S_set_H.append(V_S_set_i)
            V_S_mappings_all.append(V_S_mappings_H)
            suffix = datetime.datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
            suffix_idx = 0
            while os.path.exists(
                    os.path.join(data_folder, 'subgraph_detections_' + suffix + '_' + str(suffix_idx) + '.pkl')):
                suffix_idx += 1
            subgraph_file = os.path.join(data_folder, 'subgraph_detections_' + suffix + '_' + str(suffix_idx) + '.pkl')
            with open(subgraph_file, 'wb') as f:
                pickle.dump((V_S_mappings_H, V_S_set_H), f)
            w_sheet.write(sheet_rows, 0, parse.ListOfListsOfint2str(edge_list))
            w_sheet.write(sheet_rows, 1, subgraph_file)
            sheet_rows += 1
    wb.save(subgraph_files_list)
    if len(H_set) != 0:
        for i, graph in enumerate(graphs_ptg):
            V_S_mappings_G = []
            for j in range(len(H_set)):
                V_S_mappings_G.append(torch.tensor(V_S_mappings_all[j][i]))
            setattr(graph, 'subgraph_detections', V_S_mappings_G)
    return graphs_ptg
