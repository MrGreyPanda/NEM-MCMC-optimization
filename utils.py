import numpy as np
import random
import os

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
    """
    Computes the ancestor matrix of a given incidence matrix.

    Parameters:
    incidence (numpy.ndarray): A 2D numpy array of shape (num_s, num_e) containing the incidence matrix.

    Returns:
    ancestor_mat (numpy.ndarray): A 2D numpy array of shape (num_s, num_s) containing the ancestor matrix.
    """
    num_s = incidence.shape[0]
    incidence1 = incidence.copy()
    incidence2 = incidence.copy()
    for k in range(1, num_s):
        incidence1 = incidence1.dot(incidence)
        incidence2 += incidence1
    ancestor_mat = (incidence2 > 0).astype(int)
    return ancestor_mat


def compute_ll_ratios(n_parents, U, parent_weights, reduced_score_tables):
    """
    Computes the log-likelihood ratios for each cell in the NEM matrix.

    Returns:
    cell_log_ratios (numpy.ndarray): A 2D numpy array of shape (num_s, num_t) containing the log-likelihood ratios for each cell in the NEM matrix.
    Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
    """
    num_s = len(parent_weights)
    cell_log_ratios = U.copy()
    for i in range(num_s):        # iterate through all nodes
        for j in range(n_parents[i]):
            cell_log_ratios[i, :] += np.log(1 - parent_weights[i][j] +
                                        parent_weights[i][j] *
                                        np.exp(reduced_score_tables[i][j]))
                         
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
    
    max_vec = np.max(cell_ratios, axis=0)
    
    return sum(np.log(np.sum(np.exp(cell_ratios - max_vec), axis=0)) + max_vec) # Make numerically stable by ...(np.exp(self.cell_ratios - max_val))) + max_val

def read_csv_to_adj(pathname):
    current_dir = os.getcwd()

    # Read the network.csv file
    with open(os.path.join(current_dir, pathname), 'r') as f:
        # Read the first line to get num_s and num_e
        num_s, num_e = map(int, f.readline().strip().split(','))

        # Create an empty adjacency matrix
        adj_matrix = np.zeros((num_s, num_s), dtype=int)

        # Read the connected s-points and update the adjacency matrix
        for line in f:
            s_points = list(map(int, line.strip().split(',')))
            if len(s_points) != 2:
                break
            adj_matrix[s_points[0], s_points[1]] = 1
            
        # Read the last line to get the end nodes
        end_nodes = np.array(list(map(int, line.split(','))))
        errors = np.array(list(map(float, f.readline().strip().split(','))))
        
        return adj_matrix, end_nodes, errors, num_s, num_e
    
def transitive_reduction(adj_matrix):
    n = adj_matrix.shape[0]
    adj_mat = np.array(adj_matrix, copy=True)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j and adj_mat[i][k] and adj_mat[k][j]:
                    adj_mat[i][j] = 0
    return adj_mat

def dag_or_nem(num_s, U, parent_weights, reduced_score_tables, parents_list, n_parents, is_dag=False):
    dag_weights = parent_weights
    dag = np.zeros((num_s, num_s))
    for i in range(num_s):
        for j in range(n_parents[i]):
            dag_weights[i][j] = 1 * (parent_weights[i][j] > 0.5)
            dag[i, parents_list[i][j]] = dag_weights[i][j]
    if is_dag:
        return dag, compute_ll(compute_ll_ratios(n_parents, U, dag_weights, reduced_score_tables))
    nem = ancestor(dag)
    nem_weights = parent_weights
    for i in range(num_s):
        for j in range(n_parents[i]):
            nem_weights[i][j] = nem[i][parents_list[i][j]]
    nem_ll = compute_ll(compute_ll_ratios(n_parents, U, nem_weights, reduced_score_tables))
    return nem, nem_ll