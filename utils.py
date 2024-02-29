import numpy as np
import random
import os

def create_connection_mat(s_mat):
    dim_s = len(s_mat[0])
    connection_mat = np.zeros((dim_s, dim_s))
    for i in range(dim_s):
        connection_mat[i][i] = 1
        for j in range(dim_s):
            if s_mat[i][j] == 1:
                connection_mat[j][i] = 1
    return connection_mat

def create_real_knockdown_mat(s_mat, e_arr):
    dim_s = len(s_mat[0])
    conn_mat = create_connection_mat(s_mat)
    knockdown_mat = np.zeros((dim_s, len(e_arr)))
    for k, s_gene in enumerate(e_arr):
        for i in range(dim_s):
            if conn_mat[s_gene][i] == 1:
                knockdown_mat[i][k] = 1
    return knockdown_mat

def create_observed_knockdown_mat(knockdown_mat, alpha, beta, seed=42):
    random.seed(seed)
    pertubed_data = knockdown_mat.copy()
    for i in range(len(knockdown_mat)):
        for j in range(len(knockdown_mat[0])):
            rnd_num = random.random()
            if knockdown_mat [i][j] == 0 and rnd_num < alpha:
                pertubed_data[i][j] = 1
            elif knockdown_mat [i][j] == 1 and rnd_num < beta:
                pertubed_data[i][j] = 0
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

def initial_order_guess(observed_knockdown_mat):
    """
    Make an "educated" guess on the order of the nodes in the network.
    """
    num_s = observed_knockdown_mat.shape[0]
    sum_col = np.sum(observed_knockdown_mat, axis=1)
    order = np.arange(num_s)
    order = np.argsort(-sum_col)
    return order

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
    return sum(np.logaddexp.reduce(cell_ratios, axis=0))

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

def hamming_distance(mat_a, mat_b):
    return np.sum(np.abs(mat_a - mat_b))

# Example of how to run R code from Python
# import rpy2.robjects as robjects
# def execute_R_code(n, permy):
#     r = robjects.r
#     rcode = f"""
#         # your R code here
#         # use n and permy as input variables
#         n <- 4
#         permy <- c(2, 3, 1, 4)
#         parents <- vector("list", n)
#         nparents <- rep(NA, n) 
#         for (ii in 1:n) {{
#             parents[[ii]] <- permy[-which(permy == ii):-1]
#             nparents[ii] <- length(parents[[ii]]) # how many permissible parents it could have
#         }}
#     """
#     r(rcode)
#     permy = r['permy']
#     parents = r['parents']
#     nparents = r['nparents']
#     print(permy, parents, nparents)

def order_arr(order, unsorted_array):
    # Get the indices that would sort order
    sort_indices = np.argsort(order)

    # Sort the array along each axis, except the first one
    sorted_array = unsorted_array.copy()
    for axis in range(1, sorted_array.ndim):
        # Reshape sort_indices to be broadcastable over the required axis
        expanded_indices = np.expand_dims(sort_indices, tuple(range(axis)) + tuple(range(axis + 1, sorted_array.ndim)))
        # Applying the sort operation along the current axis
        sorted_array = np.take_along_axis(sorted_array, expanded_indices, axis=axis)

    # Sort along the first axis
    sorted_array = sorted_array[sort_indices]

    return sorted_array

def unorder_arr(perm_order, sorted_array):
    """
    Reorders the rows and columns of a matrix based on a given permutation order.

    Args:
        perm_order (array-like): The permutation order.
        sorted_matrix (array-like): The matrix to be reordered.

    Returns:
        array-like: The reordered matrix.
    """
    # Get the indices that would sort perm_order
    sort_indices = np.argsort(perm_order)

    # Get the indices to unsort (inverse of sorting)
    unsort_indices = np.argsort(sort_indices)

    # Unsort the array along each axis, except the first one
    original_array = sorted_array
    for axis in range(1, original_array.ndim):
        # Applying the unsort operation along the current axis
        original_array = np.take_along_axis(original_array, np.expand_dims(unsort_indices, axis=0), axis=axis)

    # Unsort along the first axis
    original_array = original_array[unsort_indices]

    return original_array

def min_swaps_to_match(arr1, arr2):
    # Create a mapping from elements to their indices in arr2
    index_map = {element: i for i, element in enumerate(arr2)}
    
    visited = set()  # Keep track of visited elements to avoid cycles
    swaps = 0  # Count the number of swaps
    
    for i in range(len(arr1)):
        while i not in visited and arr1[i] != arr2[i]:
            # Swap arr1[i] with the element that should be at the current position
            correct_idx = index_map[arr1[i]]
            arr1[i], arr1[correct_idx] = arr1[correct_idx], arr1[i]
            visited.add(i)
            swaps += 1
            i = correct_idx  # Continue with the new index
            
    return swaps

def get_real_order_guess(adj_mat):
    pass

def is_lower_triangular(arr):
    return np.allclose(np.tril(arr), arr)

