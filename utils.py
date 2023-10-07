import numpy as np
import pandas as pd

def create_connection_mat(s_mat, e_array):
    conn_mat = np.zeros((len(s_mat[0]), len(e_array)))
    for i, row in enumerate(s_mat):
        conn_mat[i] = np.isin(e_array, np.where(row == 1)[0])
    return conn_mat.astype(int)

def compute_scores(effect_nodes, data, A, B):
    # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
    score = np.sum(np.where(data[effect_nodes, :] == 1, 0, B), axis=0) # real 1, observe 1 -> 0. real 1 observe 0, FN-> B

    # suppose the rest does not have an effect on E-genes, therefore if we perturb them we expect to see 0
    score += np.sum(np.where(data[np.setdiff1d(np.arange(data.shape[0]), effect_nodes), :] == 0, 0, A), axis=0) #real 0, observe 0 -> 0. real 0 observe 1, FP-> A

    return score

def build_score_tbl(node, data, A, B):
    node_idx = np.where(data.index == node)[0][0]
    E_genes = data.columns
    S_genes = data.index
    
    # init score table
    score_tbl = pd.DataFrame(np.zeros((len(S_genes), len(E_genes))), index=S_genes, columns=E_genes)
    
    # compute first (base) row
    score_tbl.loc[node] = compute_scores(effect_nodes=node_idx, data=data.values, A=A, B=B)
    
    # compute the increment of score if we have additional parents
    for parents in set(S_genes) - {node}:
        parents_idx = np.where(data.index == parents)[0][0]
        s = np.where(data.iloc[parents_idx] == 0, B, -A)
        score_tbl.loc[parents] = s
    
    return score_tbl

def get_score_tables(data, A, B, nS, nE):
    all_scoretbls = {}
    # compute score tables for each S_gene
    for node in data.index:
        score_tbl = build_score_tbl(node, data, A, B)
        all_scoretbls[node] = score_tbl
    return all_scoretbls
