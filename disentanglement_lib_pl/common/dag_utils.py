import numpy as np


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
                break

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
    
    # remove the last None element and return
    return dag_layers[:-1]

def get_layer_mask(parents, children, interm_unit_dim, adj_mat):
    
    """
    In mask, rows correspond to parents and columns correspond to children
    """
    P, C = len(parents), len(children)
    
    # initialize with zeros
    mask = np.zeros(shape=(P,C * interm_unit_dim), dtype=np.float32)
    
    for i, p in enumerate(parents): 
        for j, c in enumerate(children):
            # check DAG interaction and fill-in 1's where interaction is justified according to given DAG
            # i.e if p's index is present in c's parent list
            if p in adj_mat[c]:
                
                # we have to set entries in the column associate with this parent eq to 1 
                # if current child `c` belongs to current parent `p`. We operate over 1 row (i.e. 1 parent)
                # but because interm units can have any dimension, the numbers of rows affected depends on that dimensionality
                # if interm_unit_dim = k , it means we have t incoming connections from parent to interm child so we have to 
                # set k entries eq to 1 in this row.
                row_range = range(j * interm_unit_dim, (j + 1) * interm_unit_dim)
                mask[ [i], row_range] = 1.0
    
    return mask

def get_mask_for_intermediate_to_output(interm_unit_dim, output_dim):
    
    M = np.zeros((output_dim * interm_unit_dim, output_dim), dtype=np.float32)
    R = np.array_split(range(output_dim * interm_unit_dim), output_dim) 
    C = [[o] for o in range(output_dim)]
    
    M[R,C] = 1.0
    
    return M

def get_mask_intermediate_to_intermediate(out_group_dim, in_out_groups, in_group_dim):
    """
    Generate weight matrix mask for intermediate layers on DAGInteractionLayer

    Parameters
    ----------
    out_group_dim: Number of units in a group for layer before output
    in_out_groups: Number of input and output groups
    in_group_dim: Number of units in a group for layer after input
    
    Returns
    -------
    M : 2-D array of shape (in_group_dim * in_out_groups, out_group_dim * in_out_groups) 
    representing the mask
    """

    from itertools import product
    M = np.zeros((in_out_groups * in_group_dim, out_group_dim * in_out_groups), dtype=np.float32)
    C = np.array_split(range(out_group_dim * in_out_groups), in_out_groups) 
    R = np.array_split(range(in_group_dim * in_out_groups), in_out_groups) 
    
    for r,c in zip(R,C):
        for p in product(r,c): 
            M[p] = 1.0

    return M
        