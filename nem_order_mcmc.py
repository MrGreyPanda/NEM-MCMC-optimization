import nem
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
import random

class NEMOrderMCMC():
    def __init__(self, nem, permutation_order):
        self.nem = nem
        self.num_s = nem.num_s
        self.U = nem.U
        self.score_table_list = nem.get_score_tables()
        self.get_permissible_parents(permutation_order)
        self.get_reduced_score_tables(self.score_table_list)
        self.compute_ll_ratios()
        
    def reset(self, permutation_order):
        self.get_permissible_parents(permutation_order)
        self.get_reduced_score_tables(self.score_table_list)
        self.compute_ll_ratios()
        

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
        # reduced_score_tables = [[] for _ in range(self.num_s)]
        reduced_score_tables = []
        for i in range(self.num_s):
            reduced_score_tables.append(np.array([score_table_list[i][j] for j in self.parents_list[i]]))
        self.reduced_score_tables = reduced_score_tables
    
    
    def compute_ll_ratios(self):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
       
        """

        cell_ratios = self.U
        for i in range(self.num_s):        # iterate through all nodes
            for j in range(self.n_parents[i]):
                cell_ratios[i, :] += np.log(1 - 
                                                self.parent_weights[i][j] + 
                                                self.parent_weights[i][j] * 
                                                np.exp(self.reduced_score_tables[i][j])), 
                                         
        self.cell_ratios = cell_ratios
        
        
    def compute_loglikelihood_score(self):
        """
        Computes the log-likelihood of the NEM model.
        Equivalent to equation 14 in the Abstract from Dr. Jack Kuipers.
        Returns:
        -------
        float:
            The log-likelihood of the NEM model.
        """
        max_val = np.max(self.cell_ratios)
        return sum(np.log(np.sum(np.exp(self.cell_ratios - max_val), axis=0))) + max_val
        
        
    def calculate_order_weights(self):
            """
            Calculates the order weights for each cell in the NEM model.
            Equivalent to equation 16 in the Abstract from Dr. Jack Kuipers.
            Returns:
            order_weights (numpy.ndarray): A 2D array of shape (num_s + 1, num_e) containing the order weights for each cell.
            """
            # max_val = np.max(self.cell_ratios)
            ll_ratio_sum = np.log(np.sum(np.exp(self.cell_ratios), axis=0))
            self.order_weights = np.exp(self.cell_ratios - ll_ratio_sum)
            print(np.sum(self.order_weights, axis=0)) ###### HERE IS THE CULPRIT, IT SHOULD BE ONES
            
            
            
    def calculate_local_optimum(self, i, j):
        """
        Calculates the local optimum for the given old and new weights.
        Equivalent to equation 19in the Abstract from Dr. Jack Kuipers.
        Args:
            weights: 

        Returns:
            numpy.ndarray: The local optimum.
        """
        a = self.order_weights[i] * (np.exp(self.reduced_score_tables[i][j]) - 1.0)
        b = 1.0 - self.parent_weights[i][j] * a + self.parent_weights[i][j] * (np.exp(self.reduced_score_tables[i][j]) - 1.0)
        c = a / b
        def local_ll_sum(x, re):
            return -np.sum(np.log(re * x + 1.0))
        
        
        #Using this minimizer requires to quote at least this:J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (2011), ACM Transactions on Mathematical Software, 38, 1.
        # res = fmin_l_bfgs_b(local_ll_sum, x0=0.5, bounds=[(0.1, 1.0)], args=(c,), approx_grad=True, factr=10, epsilon=1e-8, maxls=10)
        # if res[2]['warnflag'] != 0:
        #     raise Exception("Minimization not successful, Reason: {res[2]['task']}")
        
        # return res[0]
        res = minimize(local_ll_sum, x0=self.parent_weights[i][j], bounds=[(0.1, 1.0)], args=(c,), method='L-BFGS-B')
        print(res.x)
        return res.x
    
    
    
    def get_optimal_weights(self, abs_diff=0.1, max_iter = 20):
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
        ll = 0.0
        #abs_diff could be varied
        while(ll_diff > abs_diff and iter_count < max_iter):
            # print(f"Iteration: {iter_count}")
            self.compute_ll_ratios()
            self.calculate_order_weights()
            ll = self.compute_loglikelihood_score()
            
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    new_parent_weights[i][j] = self.calculate_local_optimum(i, j)
                    
            # one could vary the time of the computation of ll_ratios
            self.parent_weights = new_parent_weights
            ll_diff = ll - old_ll
            old_ll = ll
            iter_count += 1
            
        return self.parent_weights, ll
        
    def method(self, swap_prob=0.95, gamma=1, seed=42, n_iterations=500):
        permutation_order = np.array(random.sample(range(self.num_s), self.num_s))
        permutation_list = []
        score_list = []
        weights_list = []
        for i in range(n_iterations):
            print(f"##########-Iteration: {i}")
            swap = random.random() < swap_prob
            if swap:
                # swap two random nodes
                i, j = random.sample(range(self.num_s), 2)
                permutation_order[i], permutation_order[j] = permutation_order[j], permutation_order[i]
            else:
                # swap two adjacent nodes
                i = random.randint(0, self.num_s - 2)
                permutation_order[i], permutation_order[i + 1] = permutation_order[i + 1], permutation_order[i]
                
            
            result = self.get_optimal_weights()
            self.reset(permutation_order)
            score_list.append(result[1])
            weights_list.append(result[0])
            permutation_list.append(permutation_order)
        
        return score_list, weights_list, permutation_list