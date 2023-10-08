import numpy as np
import random

def create_connection_mat(s_mat):
    dim_s = len(s_mat[0])
    connection_mat = s_mat
    for i in range(dim_s):
        connection_mat[i][i] = 1
        for j in range(dim_s):
            if s_mat[i][j] == 1:
                connection_mat[j][i] = 1
    return connection_mat

def create_real_knockdown_mat(s_mat, e_arr):
    dim_s = len(s_mat[0])
    connection_mat = create_connection_mat(s_mat)
    knockdown_mat = np.zeros((dim_s, len(e_arr)))
    for k, s_gene in enumerate(e_arr):
        for i in range(dim_s):
            if connection_mat[s_gene-1][i] == 1:
                knockdown_mat[i][k] = 1
    return knockdown_mat

def create_observed_knockdown_mat(knockdown_mat, alpha, beta, seed=42):
    random.seed(seed)
    pertubed_data = knockdown_mat
    for indices, kd_observation in np.ndenumerate(knockdown_mat):
        rnd_num = random.random()
        if kd_observation == 1 and rnd_num < beta:
            pertubed_data[indices] = 0
        elif kd_observation == 0 and rnd_num < alpha:
            pertubed_data[indices] = 1
    return pertubed_data
            

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
