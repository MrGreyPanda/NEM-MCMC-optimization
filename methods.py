import numpy as np
import utils
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.linalg import solve_triangular, inv

def local_ll_sum_γ(γ, c):
    return (-np.sum(np.log(γ * c + 1.0)), -np.sum(c / (γ * c + 1.0)))

def local_ll_sum_β(β, c):
    res = -np.sum(np.log(np.exp(β) * c + 1.0))
    return res

def local_ll_sum_w(w, c):
    res = -np.sum(np.log(expit(w) * c + 1.0))
    return res

# def method(method, order, num_s, num_e, U, score_tables)

class InverseMethod:
    def __init__(self, order, num_s, num_e, U, score_tables):
        self.order = order
        self.num_s = num_s
        self.num_e = num_e
        self.U = U
        self.score_tables = score_tables
        self.eye = np.eye(self.num_s)
        self.mask = np.zeros((self.num_s, self.num_s))
        # self.mask = np.full((self.num_s, self.num_s), -5000.0)
        self.get_permissible_parents(order, self.mask, init_val=1.0)
        
        
    def get_permissible_parents(self, perm_order, weights, init_val=0.5, i1=None, i2=None):
        parents_list = np.empty(self.num_s, dtype=object)
        for i in range(self.num_s):
            index = np.where(perm_order == i)[0][0]
            parents_list[i] = perm_order[:index]
            for j in parents_list[i]:
                weights[i][j] = init_val
        self.parents_list = parents_list
        return weights
        
    def create_dag(self, weights):
        dag_weights = weights.copy()
        dag_weights = 1 * (weights > 0.5)
        return dag_weights

    def compute_cell_ratios(self, weights, score_tables):
        cell_ratios = self.U.copy()
        for i in range(self.num_s): 
            for j in self.parents_list[i]:
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
    
    def exp_parent_weights(self, weights):
        new_weights = weights.copy()
        for i in range(self.num_s):
            for j in self.parents_list[i]:
                new_weights[i][j] = np.exp(new_weights[i][j])
                
        return new_weights
    
    def local_ll_sum_b_inv(self, x, weights, i, k, local_vec, a_vec):
        weights[i][k] = x
        weights = utils.order_arr(self.order, np.exp(weights))
        B = solve_triangular(self.eye - weights, self.eye, lower=True)
        B = B / (1.0 + B)
        B = utils.unorder_arr(self.order, B)
        b_vec = 1.0 - B[i][k] * a_vec + B[i][k] * (local_vec - 1.0)
        c_vec = a_vec / b_vec
        res = -np.sum(np.log(B[i][k] * c_vec + 1.0))
        return res
    
    def calculate_local_optimum_b_inv(self, i, k, order_weights, bounds, expit_weights, weights):
        local_vec = np.exp(self.score_tables[i][k])
        a_vec = (local_vec - 1.0) * order_weights[i]
        # b_vec = 1.0 - expit_weights[i][k] * a_vec + expit_weights[i][k] * (local_vec - 1.0)
        # c_vec = a_vec / b_vec
        res = minimize(self.local_ll_sum_b_inv, x0=weights[i][k], bounds=bounds, options={'eps': 1e-2}, args=(weights, i, k, local_vec, a_vec), method='L-BFGS-B', tol=0.1)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x
    
    def opt_b(self, weights, bounds):
        new_parent_weights = utils.order_arr(self.order, weights.copy())
        inv_weights = solve_triangular(self.eye - np.exp(new_parent_weights), self.eye, lower=True)
        expit_weights = inv_weights / (1.0 + inv_weights)
        expit_weights = utils.unorder_arr(self.order, expit_weights)
        
        cell_ratios = self.compute_cell_ratios(expit_weights, self.score_tables)
        order_weights, ll = self.calculate_ll(cell_ratios)
        new_parent_weights = utils.unorder_arr(self.order, new_parent_weights)
        for i in range(self.num_s):
            for k in self.parents_list[i]:
                new_parent_weights[i][k] = self.calculate_local_optimum_b_inv(i, k, order_weights, bounds, expit_weights, new_parent_weights)
        weights = new_parent_weights.copy()
        return ll, weights
    
    def optimize(self, max_iter=1000, rel_diff=1e-8):
        bounds = [(-5000, 500)]
        init_val = 0.0
        # weights = np.zeros((self.num_s, self.num_s))
        weights = np.full((self.num_s, self.num_s), -5000.0)
        weights = self.get_permissible_parents(perm_order=self.order, weights=weights, init_val=init_val)
        ll_diff = float('inf')
        ll_old = -float('inf')
        ll_list = []
        weight_list = []
        best_ll = -float('inf')
        best_index = 0
        iter_count = 0
        while iter_count < max_iter and ll_diff > rel_diff:
            ll, weights = self.opt_b(weights, bounds)
            ll_list.append(ll)
            if ll > best_ll:
                best_ll = ll
                best_index = iter_count
            weight_list.append(weights)
            ll_diff = np.abs(ll - ll_old)
            ll_old = ll
            # print(f"Iteration {iter_count}: LL: {ll}, ll_diff: {ll_diff}")
            iter_count += 1
        weights = weight_list[best_index]
        print(f"Best ll: {best_ll}")
        weights = utils.order_arr(self.order, np.exp(weights))
        # B_tilde = inv(np.eye(self.num_s) - np.exp(weights)) - np.eye(self.num_s)
        B_tilde = solve_triangular(self.eye - weights, self.eye, lower=True)
        B_tilde = B_tilde / (1.0 + B_tilde)
        B_tilde = 1 * (B_tilde > 0.5)
        B_tilde = utils.unorder_arr(self.order, B_tilde)
        _, real_ll = self.calculate_ll(self.compute_cell_ratios(B_tilde, self.score_tables))
        print(f"Rounded LL: {real_ll}")
        return B_tilde.T, real_ll


# class ExpitMethod:
#     def __init__(self, order, num_s, num_e, U, score_tables, adj_mat):
#         self.order = order
#         self.num_s = num_s
#         self.num_e = num_e
#         self.U = U
#         self.score_tables = score_tables
#         self.eye = np.eye(self.num_s)
#         self.mask = np.zeros((self.num_s, self.num_s))
#         self.get_permissible_parents(order, self.mask, init_val=1.0)
#         self.adj_mat = np.array(adj_mat, dtype=float)
        
#     def get_permissible_parents(self, perm_order, weights, init_val=0.5, i1=None, i2=None):
#         parents_list = np.empty(self.num_s, dtype=object)
#         for i in range(self.num_s):
#             index = np.where(perm_order == i)[0][0]
#             parents_list[i] = perm_order[:index]
#             for j in parents_list[i]:
#                 weights[i][j] = init_val
#         self.parents_list = parents_list
#         return weights
        
#     def create_dag(self, weights):
#         dag_weights = weights.copy()
#         dag_weights = 1 * (weights > 0.5)
#         return dag_weights

#     def compute_cell_ratios(self, weights, score_tables):
#         cell_ratios = self.U.copy()
#         for i in range(self.num_s): 
#             for j in self.parents_list[i]:
#                 cell_ratios[i, :] += np.log(1.0 -
#                                             weights[i][j] +
#                                             weights[i][j] *
#                                             np.exp(score_tables[i][j]))
        
#         return cell_ratios

#     def calculate_ll(self, cell_ratios):
#         cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
#         order_weights = np.exp(cell_ratios - cell_sums)
#         ll = sum(cell_sums)
#         return order_weights, ll    

#     def calculate_local_optimum_w(self, i, k, order_weights, weights, bounds):
#         local_vec = np.exp(self.score_tables[i][k])
#         a = (local_vec - 1.0) * order_weights[k]
#         b = 1.0 - expit(weights[i][k]) * a + expit(weights[i][k]) * (local_vec - 1.0)
#         c = a / b
#         res = minimize(local_ll_sum_w, x0=weights[i][k], bounds=bounds, args=(c,), method='L-BFGS-B', tol=0.01)
#         if res.success is False:
#             raise Exception(f"Minimization not successful, Reason: {res.message}")
        
#         return res.x
    
#     def opt_w(self, weights, bounds):
#         cell_ratios = self.compute_cell_ratios(self.expit_parent_weights(weights), self.score_tables)
#         order_weights, ll = self.calculate_ll(cell_ratios)
#         new_parent_weights = weights.copy()
#         for i in range(self.num_s):
#             for k in self.parents_list[i]:
#                 new_parent_weights[i][k] = self.calculate_local_optimum_w(i, k, order_weights, new_parent_weights, bounds)
        
#         weights = new_parent_weights.copy()    
#         return ll, weights
    

# class ExpMethod:
#     def __init__(self, order, num_s, num_e, U, score_tables, adj_mat):
#         self.order = order
#         self.num_s = num_s
#         self.num_e = num_e
#         self.U = U
#         self.score_tables = score_tables
#         self.eye = np.eye(self.num_s)
#         self.mask = np.zeros((self.num_s, self.num_s))
#         self.get_permissible_parents(order, self.mask, init_val=1.0)
#         self.adj_mat = np.array(adj_mat, dtype=float)
        
#     def get_permissible_parents(self, perm_order, weights, init_val=0.5, i1=None, i2=None):
#         parents_list = np.empty(self.num_s, dtype=object)
#         for i in range(self.num_s):
#             index = np.where(perm_order == i)[0][0]
#             parents_list[i] = perm_order[:index]
#             for j in parents_list[i]:
#                 weights[i][j] = init_val
#         self.parents_list = parents_list
#         return weights
        
#     def create_dag(self, weights):
#         dag_weights = weights.copy()
#         dag_weights = 1 * (weights > 0.5)
#         return dag_weights

#     def compute_cell_ratios(self, weights, score_tables):
#         cell_ratios = self.U.copy()
#         for i in range(self.num_s): 
#             for j in self.parents_list[i]:
#                 cell_ratios[i, :] += np.log(1.0 -
#                                             weights[i][j] +
#                                             weights[i][j] *
#                                             np.exp(score_tables[i][j]))
        
#         return cell_ratios

#     def calculate_ll(self, cell_ratios):
#         cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
#         order_weights = np.exp(cell_ratios - cell_sums)
#         ll = sum(cell_sums)
#         return order_weights, ll
    
#     def calculate_local_optimum_β(self, i, k, order_weights, weights, bounds):
#         local_vec = np.exp(self.score_tables[i][k])
#         a = (local_vec - 1.0) * order_weights[k]
#         b = 1.0 - np.exp(weights[i][k]) * a + np.exp(weights[i][k]) * (local_vec - 1.0)
#         c = a / b
#         res = minimize(local_ll_sum_β, x0=weights[i][k], bounds=bounds, args=(c,), method='L-BFGS-B', tol=0.01)
#         if res.success is False:
#             raise Exception(f"Minimization not successful, Reason: {res.message}")
        
#         return res.x

#     def opt_β(self, weights, bounds):
#         cell_ratios = self.compute_cell_ratios(np.exp(weights), self.score_tables)
#         order_weights, ll = self.calculate_ll(cell_ratios)
#         new_parent_weights = weights.copy()
#         for i in range(self.num_s):
#             for k in self.parents_list[i]:
#                 new_parent_weights[i][k] = self.calculate_local_optimum_β(i, k, order_weights, new_parent_weights, bounds)
        
#         weights = new_parent_weights.copy()
#         return ll, weights
    
#     def optimize(self, max_iter=1000, rel_diff=1e-8):
#         bounds = [(-40, 40)]
#         init_val = 6.0
#         weights = np.zeros((self.num_s, self.num_s))
#         weights = self.get_permissible_parents(perm_order=self.order, weights=weights, init_val=init_val)
#         ll_diff = float('inf')
#         ll_old = -float('inf')
#         ll_list = []
#         weight_list = []
#         best_ll = -float('inf')
#         best_index = 0
#         iter_count = 0
#         while iter_count < max_iter and ll_diff > rel_diff:
#             ll, weights = self.opt_β(weights, bounds)
#             ll_list.append(ll)
#             if ll > best_ll:
#                 best_ll = ll
#                 best_index = iter_count
#             weight_list.append(weights)
#             ll_diff = np.abs(ll - ll_old)
#             ll_old = ll
#             iter_count += 1
#             print(f"Iteration {iter_count}: LL: {ll}, ll_diff: {ll_diff}")
            
#         weights = weight_list[best_index]
#         print(f"Best ll: {best_ll}")
#         B_tilde = inv(np.eye(self.num_s) - self.exp_parent_weights(weights)) - np.eye(self.num_s)
#         B_tilde = B_tilde / (1.0 + B_tilde) * self.mask
#         B_tilde = 1 * (B_tilde > 0.5)
#         _, real_ll = self.calculate_ll(self.compute_cell_ratios(B_tilde, self.score_tables))
#         print(f"Rounded LL: {real_ll}")
#         return B_tilde.T, real_ll
    
# ##################

# class Method:
#     def __init__(self, order, num_s, num_e, U, score_tables, adj_mat):
#         self.order = order
#         self.num_s = num_s
#         self.num_e = num_e
#         self.U = U
#         self.score_tables = score_tables
#         self.eye = np.eye(self.num_s)
#         self.mask = np.zeros((self.num_s, self.num_s))
#         self.get_permissible_parents(order, self.mask, init_val=1.0)
#         self.adj_mat = np.array(adj_mat, dtype=float)
        
#     def get_permissible_parents(self, perm_order, weights, init_val=0.5, i1=None, i2=None):
#         parents_list = np.empty(self.num_s, dtype=object)
#         for i in range(self.num_s):
#             index = np.where(perm_order == i)[0][0]
#             parents_list[i] = perm_order[:index]
#             for j in parents_list[i]:
#                 weights[i][j] = init_val
#         self.parents_list = parents_list
#         return weights
        
#     def create_dag(self, weights):
#         dag_weights = weights.copy()
#         dag_weights = 1 * (weights > 0.5)
#         return dag_weights

#     def compute_cell_ratios(self, weights, score_tables):
#         cell_ratios = self.U.copy()
#         for i in range(self.num_s): 
#             for j in self.parents_list[i]:
#                 cell_ratios[i, :] += np.log(1.0 -
#                                             weights[i][j] +
#                                             weights[i][j] *
#                                             np.exp(score_tables[i][j]))
        
#         return cell_ratios

#     def calculate_ll(self, cell_ratios):
#         cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
#         order_weights = np.exp(cell_ratios - cell_sums)
#         ll = sum(cell_sums)
#         return order_weights, ll
    
#     def calculate_local_optimum_γ(self, i, k, order_weights, weights, bounds):
#         local_vec = np.exp(self.score_tables[i][k])
#         a = (local_vec - 1.0) * order_weights[k]
#         b = 1.0 - weights[i][k] * a + weights[i][k] * (local_vec - 1.0)
#         c = a / b
#         res = minimize(local_ll_sum_γ, x0=weights[i][k], bounds=bounds, args=(c,), jac=True, method='L-BFGS-B', tol=0.01)
#         if res.success is False:
#             raise Exception(f"Minimization not successful, Reason: {res.message}")
        
#         return res.x 

#     def opt_γ(self, weights, bounds):
#         cell_ratios = self.compute_cell_ratios(weights, self.score_tables)
#         order_weights, ll = self.calculate_ll(cell_ratios)
#         new_parent_weights = weights.copy()
#         for i in range(self.num_s):
#             for k in self.parents_list[i]:
#                 new_parent_weights[i][k] = self.calculate_local_optimum_γ(i, k, order_weights, new_parent_weights, bounds)
        
#         weights = new_parent_weights.copy()
#         return ll, weights