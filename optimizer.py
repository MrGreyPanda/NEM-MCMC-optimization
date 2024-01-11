# import numpy as np
# from scipy.optimize import minimize, fmin_l_bfgs_b
# import random
# import utils
# import copy
# import wandb
# from scipy.linalg import solve_triangular, inv
# from scipy.special import expit, logit

class Optimizer:
    def __init__(self, num_s, num_e, score_tables, perm_order):
        self.num_s = num_s
        self.num_e = num_e
        self.score_tables = score_tables
        self.perm_order = perm_order
        self.parent_weights = np.zeros((num_s, num_s))
        self.W = np.zeros((num_s, num_s))
        self.cell_ratios = np.zeros((num_s, num_e))
        self.order_weights = np.zeros((num_s, num_e))
        self.iter_count = 0

    def compute_cell_ratios(self, weights, score_tables, U, num_s, parents_list):
            cell_ratios = U.copy()
            for i in range(num_s): 
                for j in parents_list[i]:
                    cell_ratios[i, :] += np.log(1.0 -
                                                weights[i][j] +
                                                weights[i][j] *
                                                np.exp(score_tables[i][j]))
            return cell_ratios

    def calculate_ll(self, cell_ratios):
        cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
        order_weights = np.exp(cell_ratios - cell_sums)
        ll = sum(cell_sums)
        return order_weights, ll

    def get_unordered_weights(self, weights, order, num_s):
        unordered_weights = np.zeros_like(weights)
        for i in range(num_s):
                index = np.where(order == i)[0]
                unordered_weights[i] = weights[index]
        return unordered_weights
        
    def get_ordered_weigths(self, weights, order, num_s):
        ordered_weights = np.zeros_like(weights)
        for i in range(num_s):
            ordered_weights[order[i]] = weights[i]
        return ordered_weights
        
    def local_opt_fun(self, Beta_tilde, parents_list, perm_order):
        cell_ratios = self.compute_cell_ratios(Beta_tilde, self.score_tables)
        order_weights, _ = self.calculate_ll(cell_ratios)
        for i in range(self.num_s):
            for k in parents_list[i]:
                local_vec = np.exp(self.score_tables[i][k])
                a_ik = (local_vec - 1.0) * order_weights[k]
                C[i][k] = a_ik 
        C_tilde = get_unordered_weights(C, perm_order)
        x = x.reshape(C_tilde.shape[0], C_tilde.shape[1])
        x = expit(inv(I - x))
        return -np.sum(np.log(x[:,:, np.newaxis] * C_tilde + 1.0))

    def opt_weights(self, max_iter=50):
        old_ll = -float('inf')
        ll_diff = float('inf')
        iter_count = 0
        ll = 0.0
        Beta_tilde = parent_weights = np.zeros((self.num_s, self.num_s))
        C = np.zeros((self.num_s, self.num_s, num_e))
        Beta_tilde = minimize(local_opt_fun, x0=Beta_tilde.flatten(), args=(), method='L-BFGS-B', tol=0.01).x.reshape((self.num_s, self.num_s))
        W_tilde = expit(inv(I - Beta_tilde))
        print(f"W_tilde: {W_tilde}")
        W = get_ordered_weigths(W_tilde, self.perm_order)
        W = 1 * (W > 0.5)
        parent_weights = W
        cell_ratios = compute_cell_ratios(W, self.score_tables)
        order_weights, ll = calculate_ll()
        ll_diff = ll - old_ll
        old_ll = ll
        while iter_count < max_iter or ll_diff < 0.01:
            print(f"Iteration of weight optimization: {iter_count}")
            Beta_tilde = minimize(local_opt_fun, x0=Beta_tilde.flatten(), args=(C_tilde, I,), method='L-BFGS-B', tol=0.01).x.reshape((self.num_s, self.num_s))
            W_tilde = expit(inv(I - Beta_tilde))
            W = get_ordered_weigths(W_tilde, self.perm_order)
            W = 1 * (W > 0.5)
            parent_weights = W
            print(f"W:\n{W}")
            cell_ratios = compute_cell_ratios(parent_weights, self.score_tables)
            order_weights, ll = calculate_ll()
            ll_diff = ll - old_ll
            old_ll = ll
            print(f"LL: {ll}")
            iter_count += 1
        return ll

import numpy as np

def order_arr(perm_order, unsorted_arr):
    # Get the indices that would sort perm_order
    sort_indices = np.argsort(perm_order)

    # Sort the array along each axis, except the first one
    sorted_arr = unsorted_arr
    for axis in range(1, sorted_arr.ndim):
        # Applying the sort operation along the current axis
        sorted_arr = np.take_along_axis(sorted_arr, np.expand_dims(sort_indices, axis=0), axis=axis)

    # Sort along the first axis
    sorted_arr = sorted_arr[sort_indices]

    return sorted_arr

def unorder_arr(perm_order, sorted_arr):
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

def main():
    perm_order = np.array([3,4,2,1,5])
    mat = np.array([
        [0,0,0,0,0],
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,1,1,0,0],
        [1,1,1,1,0]
    ])
    
    old_mat = order_mat(perm_order, mat)
    print(old_mat)
    new_mat = unorder_mat(perm_order, old_mat)
    print(new_mat)
    
if __name__ == "__main__":
    main()