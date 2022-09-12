import numpy as np
import torch

def get_adj_mat_from_adj_list(adjacency_list, return_type="torch"):
    """
    type: 'torch' or 'numpy'
    """
    num_nodes = len(adjacency_list)

    # initialize with self-connections
    A = np.zeros(shape=(num_nodes, num_nodes)) + np.eye(num_nodes)

    for node_idx, parent_list in enumerate(adjacency_list):
        for parent_node_idx in parent_list:
            A[node_idx, parent_node_idx] = 1.0

    return torch.from_numpy(A).type(torch.FloatTensor) if return_type == 'torch' else A

def adjust_adj_mat_for_prior(A):
    
    num_neighbours = A.sum(dim=-1, keepdims=True)
    A_adj = A.clone()

    for node in range(A_adj.shape[0]):
        if num_neighbours[node] > 1:
            # if the node has parents, remove the self-connection
            A_adj[node][node] = 0
    
    return A_adj.numpy()

def extend_adj_mat_with_indept_nodes(adjacency_matrix, num_dept_nodes, num_indept_nodes):
        """
        Takes is an adj. mat. A and number of independent nodes to add to it and returns 
        a new adj mat of the form
        [A, 0,
         0, I ]
        This allows us to model latents for which we don't have explicit labals / connections etc as indept latents
        """
        
        zeros_upper_right = torch.zeros(size=(num_dept_nodes, num_indept_nodes))
        zeros_lower_left = torch.zeros(size=(num_indept_nodes, num_dept_nodes))
        I = torch.eye(num_indept_nodes)

        upper_rows = torch.cat([adjacency_matrix, zeros_upper_right], dim=1) # along columns
        lower_rows = torch.cat([zeros_lower_left, I], dim=1) # along rows

        extended_adj_mat = torch.cat([upper_rows, lower_rows], dim=0) # along rows
        
        return extended_adj_mat

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
        