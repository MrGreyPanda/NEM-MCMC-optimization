import numpy as np
import pandas as pd

def create_connection_mat(s_mat, e_array):
    conn_mat = np.zeros((len(s_mat[0]), len(e_array)))
    for i, row in enumerate(s_mat):
        conn_mat[i] = np.isin(e_array, np.where(row == 1)[0])
    return conn_mat.astype(int)

# def compute_scores(effect_nodes, data, A, B):
#     """
#     Computes the scores for a given set of effect nodes, data, and parameters A and B.

#     Parameters:
#     effect_nodes (numpy.ndarray): The set of effect nodes.
#     data (numpy.ndarray): The data to compute the scores on.
#     A (float): The parameter A.
#     B (float): The parameter B.

#     Returns:
#     numpy.ndarray: The computed scores.
#     """
    
#     # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
#     score = np.sum(np.where(data[effect_nodes, :] == 1, 0, B), axis=0) # real 1, observe 1 -> 0. real 1 observe 0, FN-> B

#     # suppose the rest does not have an effect on E-genes, therefore if we perturb them we expect to see 0
#     score += np.sum(np.where(data[np.setdiff1d(np.arange(data.shape[0]), effect_nodes), :] == 0, 0, A), axis=0) #real 0, observe 0 -> 0. real 0 observe 1, FP-> A

#     return score

# def build_score_table(node, data, A, B):
#     """
#     Builds a score table for a given node in a network.

#     Parameters:
#     node (str): The node for which to build the score table.
#     data (pandas.DataFrame): The data matrix containing the gene expression data.
#     A (float): The activation parameter.
#     B (float): The inhibition parameter.

#     Returns:
#     pandas.DataFrame: The score table for the given node.
#     """
#     node_idx = np.where(data.index == node)[0][0]
#     E_genes = data.columns
#     S_genes = data.index
    
#     # init score table
#     score_tbl = pd.DataFrame(np.zeros((len(S_genes), len(E_genes))), index=S_genes, columns=E_genes)
    
#     # compute first (base) row
#     score_tbl.loc[node] = compute_scores(effect_nodes=node_idx, data=data.values, A=A, B=B)
    
#     # compute the increment of score if we have additional parents
#     for parents in set(S_genes) - {node}:
#         parents_idx = np.where(data.index == parents)[0][0]
#         s = np.where(data.iloc[parents_idx] == 0, B, -A)
#         score_tbl.loc[parents] = s
    
#     return score_tbl

# def get_score_tables(data, A, B):
#     """
#     Computes score tables for each S_gene in the given data using the given A and B matrices.
    
#     Args:
#     - data: pandas DataFrame containing the gene expression data
#     - A: numpy array representing the A matrix
#     - B: numpy array representing the B matrix
    
#     Returns:
#     - all_scoretbls: dictionary containing the score tables for each S_gene
#     """
#     all_scoretbls = {}
#     # compute score tables for each S_gene
#     for node in data.index:
#         score_tbl = build_score_table(node, data, A, B)
#         all_scoretbls[node] = score_tbl
#     return all_scoretbls


# def get_node_lr_tbl(all_score_tbls, data, A):
#     """
#     Returns a table with the node scores for all effect nodes in the network.
    
#     Parameters:
#     all_score_tbls (dict): A dictionary containing score tables for all genes in the network.
#     data (numpy.ndarray): A 2D numpy array containing the gene expression data.
#     A (float): The activation threshold for the effect nodes.
    
#     Returns:
#     pandas.DataFrame: A table with the node scores for all effect nodes in the network.
#     """
#     l = []
#     for S_gene in all_score_tbls:
#         l.append(all_score_tbls[S_gene].loc[S_gene]) # return base row

#     res = pd.concat(l, axis=1).T
#     res.index = all_score_tbls.keys()

#     # all effect nodes are disconnected (we expect their state to stay 0)
#     res.loc['S0'] = np.where(data == 0, 0, A).sum(axis=0)
    
#     return res
