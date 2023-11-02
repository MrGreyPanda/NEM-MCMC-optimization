import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
import random
import logging
import utils

def local_ll_sum(x, c):
    res = -np.sum(np.log(c * x + 1.0))
    return res

class NEMOrderMCMC():
    def __init__(self, nem, permutation_order):
        self.nem = nem
        self.num_s = nem.num_s
        self.U = nem.U
        self.score_table_list = nem.get_score_tables()
        self.get_permissible_parents(permutation_order)
        self.get_reduced_score_tables(self.score_table_list)
        self.cell_ratios = self.compute_ll_ratios()
        
    def reset(self, permutation_order):
        self.ll = 0.0
        self.get_permissible_parents(permutation_order)
        self.get_reduced_score_tables(self.score_table_list)
        # self.compute_ll_ratios()
        
        

    def get_permissible_parents(self, permutation_order):
        """
        Initializes the permissible parents and their weights for each node in the network given a permutation order.
        
        Args:
        - permutation_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.
        
        """
        parent_weights = np.empty(self.num_s, dtype=object)
        parents_list = np.empty(self.num_s, dtype=object)
        n_parents = np.empty(self.num_s, dtype=int)
        for i in range(self.num_s):
            index = np.where(permutation_order == i)[0][0]
            parents_list[i] = permutation_order[index + 1:]            
            n_parents[i] = len(parents_list[i])
            parent_weights[i] = [0.5] * n_parents[i]
        self.parents_list, self.n_parents, self.parent_weights = parents_list, n_parents, parent_weights

    def get_reduced_score_tables(self, score_table_list):
        """
        Initializes a list of reduced score tables based on the given score_table_list.

        Args:
        score_table_list (list(np.array)): a list containing the score tables for each gene in the network.
        """
        self.reduced_score_tables = []
        self.reduced_score_tables.clear()
        for i in range(self.num_s):
            self.reduced_score_tables.append(np.array([score_table_list[i][j] for j in self.parents_list[i]]))
    
    
    def compute_ll_ratios(self):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
       
        """
        cell_ratios = self.U.copy()
        for i in range(self.num_s):        # iterate through all nodes
            for j in range(self.n_parents[i]):
                cell_ratios[i, :] += np.log(1.0 - 
                                            self.parent_weights[i][j] + 
                                            self.parent_weights[i][j] * 
                                            np.exp(self.reduced_score_tables[i][j]))
        return cell_ratios
        
    def calculate_ll(self):
        max_val = np.max(self.cell_ratios, axis=0)
        cell_sums = np.log(np.sum(np.exp(self.cell_ratios - max_val), axis=0)) + max_val
        order_weights = np.exp(self.cell_ratios - cell_sums)
        ll = sum(cell_sums)
        return order_weights, ll
            
    def calculate_local_optimum(self, i, j):
        """
        Calculates the local optimum for the given old and new weights.
        Equivalent to equation 19in the Abstract from Dr. Jack Kuipers.
        Args:
            weights: 

        Returns:
            numpy.ndarray: The local optimum.
        """
        local_vec = np.exp(self.reduced_score_tables[i][j])
        a = (local_vec - 1.0) * self.order_weights[i]
        b = 1.0 - self.parent_weights[i][j] * a + self.parent_weights[i][j] * (local_vec - 1.0)
        c = a / b
        #Using this minimizer requires to quote at least this:J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        # res = fmin_l_bfgs_b(local_ll_sum, x0=0.5, bounds=[(0.1, 1.0)], args=(c,), approx_grad=True, factr=10, epsilon=1e-8, maxls=8)
        # if res[2]['warnflag'] != 0:
        #     raise Exception(f"Minimization not successful, Reason: {res[2]['task']}")
        # return res[0]
        
        res = minimize(local_ll_sum, x0=0.5, bounds=[(0.001, 1.0)], args=(c,), method='L-BFGS-B', tol=0.1)
        if res.success == False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x
    
    
    def get_optimal_weights(self, abs_diff=0.1, max_iter = 100):
        """
            Calculates the optimal weights for the NEM model using the specified relative error and maximum number of iterations.

            Args:
            - abs_diff: a float representing the absolute difference threshold for convergence (default: 0.1)
            - max_iter: an integer representing the maximum number of iterations (default: 1000)

            Returns:
            - parent_weights: a list of length num_s containing the optimal weights for each variable's parents
            
            TODO:
            - Check if log-likelihood ratios are updated accordingly
            - Tackle overflows
        """
        old_ll = -float('inf')
        ll_diff = float('inf')
        iter_count = 0
        self.ll = 0.0
        #abs_diff could be varied
        while(ll_diff > abs_diff and iter_count < max_iter):
            # print(f"Iteration: {iter_count}")
            self.cell_ratios = self.compute_ll_ratios()
            # self.order_weights = self.calculate_order_weights()
            # self.ll = self.compute_loglikelihood_score()
            
            self.order_weights, self.ll = self.calculate_ll()
            logging.info(f"self.ll: {self.ll}")
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    # self.parent_weights[i][j] = self.calculate_local_optimum(i, j)
                    new_parent_weights[i][j] = self.calculate_local_optimum(i, j)
                    
            self.parent_weights = new_parent_weights
            print(f"ll: {self.ll}")
            ll_diff = self.ll - old_ll
            old_ll = self.ll
            iter_count += 1
        return self.ll
    
    def dag_or_nem(self, is_dag = False):
        dag_weights = self.parent_weights
        dag = np.zeros((self.num_s, self.num_s))
        for i in range(self.num_s):
            for j in range(self.n_parents[i]):
                dag_weights[i][j] = 1 * (self.parent_weights[i][j] > 0.5)
                dag[i, self.parents_list[i][j]] = dag_weights[i][j]
        # dag_ll = utils.compute_ll(utils.compute_ll_ratios(self.n_parents, self.U, dag_weights, self.reduced_score_tables))
        if is_dag:
            return dag, utils.compute_ll(utils.compute_ll_ratios(self.n_parents, self.U, dag_weights, self.reduced_score_tables))
        nem = utils.ancestor(dag)
        nem_weights = self.parent_weights
        for i in range(self.num_s):
            for j in range(self.n_parents[i]):
                nem_weights[i][j] = nem[self.parents_list[i][j]][i]
        nem_ll = utils.compute_ll(utils.compute_ll_ratios(self.n_parents, self.U, nem_weights, self.reduced_score_tables))
        return nem, nem_ll
            
        
    def method(self, swap_prob=0.95, gamma=1, seed=0, n_iterations=500, is_dag=False):
        best_score = -float('inf')
        random.seed(seed)
        current_permutation_order = np.array(random.sample(range(self.num_s), self.num_s))
        current_score = self.get_optimal_weights()
        permutation_order = current_permutation_order
        best_nem = []
        for i in range(n_iterations):
            print(f"Starting MCMC iteration {i}")
            print()
            logging.info(f"Starting MCMC iteration {i}")
            logging.info(f"###########################")
            swap = random.random() < swap_prob
            
            if swap:
                # swap two random nodes
                i, j = random.sample(range(self.num_s), 2)
                permutation_order[i], permutation_order[j] = permutation_order[j], permutation_order[i]
            else:
                # swap two adjacent nodes
                i = random.randint(0, self.num_s - 2)
                permutation_order[i], permutation_order[i + 1] = permutation_order[i + 1], permutation_order[i]
                
            score = self.get_optimal_weights()
            
            acceptance_rate = np.exp(gamma * abs(score - current_score))
            if random.random() < acceptance_rate:
                current_permutation_order = permutation_order
                current_score = score
                if(current_score > best_score):
                    best_score = current_score
                    best_weights = self.parent_weights
                    best_parents = self.parents_list
                    ##nem
                    dag_weights = self.parent_weights
                    dag = np.zeros((self.num_s, self.num_s))
                    for i in range(self.num_s):
                        for j in range(self.n_parents[i]):
                            dag_weights[i][j] = 1 * (self.parent_weights[i][j] > 0.5)
                            dag[i, self.parents_list[i][j]] = dag_weights[i][j]
                    best_nem = utils.ancestor(dag)   
            self.reset(permutation_order)

        return best_score, best_nem
    