from more_itertools import first
import numpy as np
import seaborn, time
import pandas as pd
from pomegranate import BayesianNetwork

def find_top_level_nodes(adj_mat):
    
    first_layer = []
    
    for node_idx, node in enumerate(adj_mat):
        # if in-degree is 0 add it to first layer node
        if len(node) == 0:
            first_layer.append(node_idx)
    
    return None if len(first_layer) == 0 else first_layer  

def find_child_nodes(parent_nodes, adj_mat):

    child_nodes = []
    for node_idx, node in enumerate(adj_mat):
        for p in parent_nodes:
            if p in node:
                child_nodes.append(node_idx)

    return None if len(child_nodes) == 0 else child_nodes  

def get_dag_layers(adj_mat):
    
    dag_layers = []
    first_layer = find_top_level_nodes(adj_mat)
    dag_layers.append(first_layer)
    
    children, parents = [], first_layer
    while children is not None:
        children = find_child_nodes(parents, adj_mat)
        dag_layers.append(children)
        parents = children

def get_layer_mask(parents, children, adj_mat):
    
    """
    In mask, rows correspond to parents and columns correspond to children
    """
    P, C = len(parents), len(children)
    
    # initialize with zeros
    mask = np.zeros(shape=(P,C))
    
    for i, p in enumerate(parents): 
        for j, c in enumerate(children):
            # check DAG interaction and fill-in 1's where interaction is justified according to given DAG
            # i.e if p's index is present in c's parent list
            if p in adj_mat[c]:
                mask[i, j] = 1.0
    
    return mask