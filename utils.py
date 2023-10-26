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

def ancestor(incidence):
    incidence1 = incidence
    incidence2 = incidence
    k = 1
    while k < incidence.shape[0]:
        incidence1 = incidence1.dot(incidence)
        incidence2 = incidence2 + incidence1
        k += 1
    incidence2[incidence2 > 0] = 1
    return incidence2


def compute_ll_ratios(parent_lists, U, parent_weights, reduced_score_tables):
    """
    Computes the log-likelihood ratios for each cell in the NEM matrix.

    Returns:
    cell_log_ratios (numpy.ndarray): A 2D numpy array of shape (num_s, num_t) containing the log-likelihood ratios for each cell in the NEM matrix.
    Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
    """
    num_s = len(parent_lists[0])
    nparents = [len(parent_lists[i]) for i in range(num_s)]
    cell_log_ratios = U
    for i in range(num_s):        # iterate through all nodes
        if nparents[i] > 1:
            cell_log_ratios[i, :] += np.sum(np.log(1 - parent_weights[i][:, np.newaxis] +
                                            parent_weights[i][:, np.newaxis] *
                                            np.exp(reduced_score_tables[i])),
                                    axis=0)
    return cell_log_ratios

def compute_ll(cell_ratios):
    """
    Computes the log-likelihood of the NEM model.

    Returns: the total log-likelihood score of the given order.
    Equivalent to equation 14 in the Abstract from Dr. Jack Kuipers.
    -------
    float:
        The log-likelihood of the NEM model.
    """
    # cellLRs_t = cellLRs.T # us this in the sum
    # max_val = np.max(cellLRs, axis=0)
    return sum(np.log(np.sum(np.exp(cell_ratios), axis=0))) # Make numerically stable by ...(np.exp(self.cell_ratios - max_val))) + max_val

