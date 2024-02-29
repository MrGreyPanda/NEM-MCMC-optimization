import numpy as np
import utils
import os
import random
from scipy.optimize import minimize


class NEM:
    def __init__(self, adj_matrix, end_nodes, errors, num_s, num_e, seed=42):
        self.num_s = num_s
        self.num_e = num_e
        self.adj_matrix = adj_matrix
        alpha = errors[0]
        beta = errors[1]
        self.real_knockdown_mat = utils.create_real_knockdown_mat(adj_matrix, end_nodes)
        random.seed(seed)
        self.A = np.log(alpha / (1.0 - beta))
        self.B = np.log(beta / (1.0 - alpha))
        self.observed_knockdown_mat = utils.create_observed_knockdown_mat(self.real_knockdown_mat, alpha, beta)
        self.U = self.get_node_lr_table(self.get_score_tables(self.observed_knockdown_mat))
        self.real_order_ll, self.real_ll, self.real_parent_order = self.compute_real_score(real_knockdown_mat=self.real_knockdown_mat, adj_mat=adj_matrix)
        self.obs_order_ll, self.obs_ll, self.obs_parent_order = self.compute_real_score(real_knockdown_mat=self.observed_knockdown_mat, adj_mat=adj_matrix)
        
        
    def compute_scores(self, node, knockdown_mat):
        # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
        score = np.where(knockdown_mat[node, :] == 1, 0, self.B)  # real 1, observe 1 -> 0. real 1 observe 0, FN-> B

        # suppose the rest does not have an effect on E-genes, therefore if we perturb them we expect to see 0
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            score += np.where(knockdown_mat[index, :] == 1, self.A, 0)  # real 0, observe 0 -> 0. real 0 observe 1, FP-> A

        return score
    
    def build_score_table(self, node, knockdown_mat):
        # init score table
        score_table = np.zeros((self.num_s, self.num_e))
        
        # compute first (base) row
        score_table[node, :] = self.compute_scores(node, knockdown_mat)
        
        # compute the increment of score if we have additional parents
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            score_table[index, :] = np.where(knockdown_mat[index] == 0, self.B, -self.A)
        return score_table
    
    def get_score_tables(self, knockdown_mat):
        all_score_tables = []
        # compute score tables for each S_gene
        for node in range(self.num_s):
            all_score_tables.append(self.build_score_table(node, knockdown_mat))
        return all_score_tables

    def get_node_lr_table(self, all_score_tables):
        l = []
        for i in range(self.num_s):
            l.append(all_score_tables[i][i])  # return base row
            
        res = np.vstack(l)
        res = np.vstack((res, np.where(self.observed_knockdown_mat == 0, 0, self.A).sum(axis=0)))
        
        return res
    
    def get_reduced_score_tables(self, score_table_list, parents_list):
        reduced_score_tables = []
        for i in range(self.num_s):
            reduced_score_tables.append(np.array([score_table_list[i][j] for j in parents_list[i]]))
        return reduced_score_tables
    
    def compute_ll_ratios(self, parent_weights, reduced_score_tables, real_n_parents, U):
        cell_ratios = U.copy()
        for i in range(self.num_s):        # iterate through all nodes
            for j in range(real_n_parents[i]):
                cell_ratios[i, :] += np.log(1.0 - 
                                            parent_weights[i][j] + 
                                            parent_weights[i][j] * 
                                            np.exp(reduced_score_tables[i][j]))
        return cell_ratios
    
    def calculate_ll(self, cell_ratios):
        cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
        order_weights = np.exp(cell_ratios - cell_sums)
        ll = sum(cell_sums)
        return order_weights, ll
    
    def compute_real_score(self, real_knockdown_mat, adj_mat):
        # Implement the real nem Score, not only the order score ( Order is good for comparison)
        for i in range(self.num_s):
            adj_mat[i][i] = 0
        row_sums = np.sum(adj_mat, axis=1)
        sorted_indices = np.argsort(row_sums)[::-1]
        parent_order = sorted_indices
        score_table_list = self.get_score_tables(real_knockdown_mat)
        real_U = self.get_node_lr_table(score_table_list)
        n_parents = np.empty(self.num_s, dtype=int)
        parents_list = np.empty(self.num_s, dtype=object)
        real_parent_weights = np.empty(self.num_s, dtype=object)
        for i in range(self.num_s):
            index = np.where(parent_order == i)[0][0]
            parents_list[i] = parent_order[:index]
            n_parents[i] = len(parents_list[i])
            real_parent_weights[i] = [0.5] * n_parents[i]
        reduced_score_table = self.get_reduced_score_tables(score_table_list, parents_list)
        ll_diff = float('inf')
        old_ll = -float('inf')
        abs_diff = 0.0001
        iter_count = 0
        max_iter = 1000
        parent_weights = real_parent_weights.copy()
        while ll_diff > abs_diff and iter_count < max_iter:
            order_weights, ll = self.calculate_ll(self.compute_ll_ratios(real_parent_weights, reduced_score_table, n_parents, real_U))
            for i in range(self.num_s):
                for j in range(n_parents[i]):
                    local_vec = np.exp(reduced_score_table[i][j])
                    a = (local_vec - 1.0) * order_weights
                    b = 1.0 - real_parent_weights[i][j] * a + real_parent_weights[i][j] * (local_vec - 1.0)
                    c = a / b
                    res = minimize(local_ll_sum, x0=0.5, bounds=[(0.0, 1.0)], args=(c,), method='L-BFGS-B', tol=0.1)
                    parent_weights[i][j] = res.x
            real_parent_weights = parent_weights
            ll_diff = ll - old_ll
            old_ll = ll
            iter_count += 1
        
        for i in range(self.num_s):
            for j in range(n_parents[i]):
                real_parent_weights[i][j] = 1 * (real_parent_weights[i][j] > 0.5)
        _, real_order_ll = self.calculate_ll(self.compute_ll_ratios(real_parent_weights, reduced_score_table, n_parents, real_U))
        
        parents_list = [[] for _ in range(self.num_s)]
        real_parent_weights =[[] for _ in range(self.num_s)]
        n_parents = np.zeros(self.num_s, dtype=int)
        for i in range(self.num_s):
            for j in range(self.num_s):
                if adj_mat[i][j] == 1:
                    parents_list[j].append(i)
                    n_parents[j] += 1
                    real_parent_weights[j].append(1.0)
        reduced_score_table = self.get_reduced_score_tables(score_table_list, parents_list)
        _, real_ll = self.calculate_ll(self.compute_ll_ratios(real_parent_weights, reduced_score_table, n_parents, real_U))
                    
        return real_order_ll, real_ll, parent_order
    
    # Include grandparents, grandgrandparents... into the optimization
    # Run on multiple examples to see what to do next (what to optimize)
    
    
    # Possible optimizations:
    # - use some sort of clustering on w to find similarities, as some measurements should be linked
    #   Use the clustering for shared probablity between clusters for the error rates
    #   use that as a latent variable
    
def local_ll_sum(x, c):
    res = -np.sum(np.log(c * x + 1.0))
    return res