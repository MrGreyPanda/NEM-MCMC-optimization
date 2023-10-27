import numpy as np
import utils
import os
import random
from scipy.optimize import minimize

class NEM:
    """
    A class representing a Network Error Model (NEM).

    Attributes:
    - num_s (int): the number of s-points in the network
    - num_e (int): the number of e-points in the network
    - e_arr (numpy.ndarray): the end nodes of the network
    - A (float): the value of the parameter A in the NEM
    - B (float): the value of the parameter B in the NEM
    - observed_knockdown_mat (numpy.ndarray): the connection matrix of the network
    - parents_list (list(int)): list of parents of node i
    - n_parents (list(int)): number of parents node i has
    - parent_weights (list(np.array(float))): the weights of the parents of node i
    

    Methods:
    - __init__(self): initializes the NEM object by reading the network.csv file and setting the attributes
    """


    def __init__(self, adj_matrix, end_nodes, errors, num_s, num_e):
        """
        Initializes the NEM object by reading the network.csv file and setting the attributes.
        
        :param pathname: The path to the network.csv file.
        :type pathname: str
        
        :return: None
        """
        self.num_s = num_s
        self.num_e = num_e
        
        alpha = errors[0]
        beta = errors[1]
        real_knockdown_mat = utils.create_real_knockdown_mat(adj_matrix, end_nodes)
        permutation_order = np.array(random.sample(range(self.num_s), self.num_s))
        
        self.A = np.log(alpha / (1.0 - beta))
        self.B = np.log(beta / (1.0 - alpha))
        self.observed_knockdown_mat = utils.create_observed_knockdown_mat(real_knockdown_mat, alpha, beta)
        # score_table_list = get_score_tables()
        self.get_permissible_parents(permutation_order)
        self.U = self.get_node_lr_table(self.get_score_tables())
        self.reduced_score_tables = self.get_reduced_score_tables(self.get_score_tables())
        self.compute_ll_ratios()
    
        
    def compute_scores(self, node):
        """
        Computes the scores for a given set of effect nodes, data, and parameters A and B.
        Returns:
        numpy.ndarray: The computed scores.
        """
        # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
        score = np.where(self.observed_knockdown_mat[node, :] == 1, 0, self.B) # real 1, observe 1 -> 0. real 1 observe 0, FN-> B

        # suppose the rest does not have an effect on E-genes, therefore if we perturb them we expect to see 0
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            score += np.where(self.observed_knockdown_mat[index, :] == 1, self.A, 0) #real 0, observe 0 -> 0. real 0 observe 1, FP-> A

        return score
    
    def get_permissible_parents(self, permutation_order):
        parent_weights = np.empty(self.num_s, dtype=object)
        parents_list = np.empty(self.num_s, dtype=object)
        n_parents = np.empty(self.num_s, dtype=int)
        for i in range(self.num_s):
            index = np.where(permutation_order == i)[0][0]
            parents_list[i] = permutation_order[index + 1:]            
            n_parents[i] = len(parents_list[i])
            parent_weights[i] = [0.5] * n_parents[i]
        
        self.parents_list, self.n_parents, self.parent_weights = parents_list, n_parents, parent_weights
    
    def build_score_table(self, node):
        """
        Builds a score table for a given node in a network.

        Parameters:
        node (int): The node for which to build the score table.
        data (pandas.DataFrame): The data matrix containing the gene expression data.
 
        Returns:
        pandas.DataFrame: The score table for the given node.
        """
        # init score table
        score_table = np.zeros((self.num_s, self.num_e))
        
        # compute first (base) row
        score_table[node, :] = self.compute_scores(node)
        
        # compute the increment of score if we have additional parents
        
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            s = np.where(self.observed_knockdown_mat[index] == 0, self.B, -self.A)
            score_table[index] = s
        
        return score_table
    
    def get_score_tables(self):
        """
        Computes score tables for each S_gene in the given data using the given A and B matrices.
        
        Args:
        - data: pandas DataFrame containing the gene expression data
        - A: numpy array representing the A matrix
        - B: numpy array representing the B matrix
        
        Returns:
        - all_scoretbls: list containing the score tables for each S_gene
        """
        all_score_tables = []
        # compute score tables for each S_gene
        for node in range(self.num_s):
            all_score_tables.append(self.build_score_table(node))
        return all_score_tables
        

    def get_node_lr_table(self, all_score_tables):
        """
        Returns a table with the node scores for all effect nodes in the network.
        
        Parameters:
        all_score_tables (list): A list containing score tables for all genes in the network.
        data (numpy.ndarray): A 2D numpy array containing the gene expression data.
        A (float): The activation threshold for the effect nodes.
        
        Returns:
        numpy.ndarray: A table with the node scores for all effect nodes in the network.
        """
        l = []
        for i in range(self.num_s):
            l.append(all_score_tables[i][i]) # return base row

        res = np.vstack(l)
        res = np.vstack((res, np.where(self.observed_knockdown_mat == 0, 0, self.A).sum(axis=0)))

        return res
    
    def get_reduced_score_tables(self, score_table_list):
        """
        Returns a list of reduced score tables based on the given parent lists.

        Args:
        parentLists (list): A list of lists containing the indices of the parents for each node.

        Returns:
        list: A list of lists of numpy arrays containing the scores of each node's parents.
        """
        # reduced_score_tables = [[] for _ in range(self.num_s)]
        reduced_score_tables = []
        for i in range(self.num_s):
            reduced_score_tables.append(np.array([score_table_list[i][j] for j in self.parents_list[i]]))
        return reduced_score_tables
    
    
    def compute_ll_ratios(self):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
        Returns:
        cellLRs (numpy.ndarray): A 2D numpy array of shape (num_s, num_t) containing the log-likelihood ratios for each cell in the NEM matrix.
        """

        cell_ratios = self.U
        for i in range(self.num_s):        # iterate through all nodes
            for j in range(self.n_parents[i]):
                cell_ratios[i, :] += np.sum(np.log(1. - 
                                                self.parent_weights[i][j] + 
                                                self.parent_weights[i][j] * 
                                                np.exp(self.reduced_score_tables[i][j])), 
                                         axis=0)
        self.cell_ratios = cell_ratios
    
    def compute_likelihood_score(self):
        """
        Computes the log-likelihood of the NEM model.
        Equivalent to equation 14 in the Abstract from Dr. Jack Kuipers.
        Returns:
        -------
        float:
            The log-likelihood of the NEM model.
        """
        # cellLRs_t = cellLRs.T # us this in the sum
        max_val = np.max(self.cell_ratios)
        # sum_res = np.sum(np.exp(self.cell_ratios - max_val), axis=0)
        # logar = np.log(sum_res)
        # print(f"Logorithm: {logar}")
        # print(f"sum: {sum_res}")
        
        return sum(np.log(np.sum(np.exp(self.cell_ratios - max_val), axis=0))) + max_val# Make numerically stable by ...(np.exp(self.cell_ratios - max_val))) + max_val
        
        
    def calculate_order_weights(self):
            """
            Calculates the order weights for each cell in the NEM model.
            Equivalent to equation 16 in the Abstract from Dr. Jack Kuipers.
            Returns:
            order_weights (numpy.ndarray): A 2D array of shape (num_s + 1, num_e) containing the order weights for each cell.
            """
            # max_val = np.max(self.cell_ratios)
            ll_ratio_sum = np.sum(np.exp(self.cell_ratios), axis=0)
            
            # order_weights = np.zeros((self.num_s + 1, self.num_e))

            # for i in range(self.num_s + 1):
            #     order_weights[i, :] = np.exp(self.cell_ratios[i, :]) / ll_ratio_sum
            order_weights = np.exp(self.cell_ratios) / ll_ratio_sum
            
            self.order_weights = order_weights
            
            
    def calculate_local_optimum(self, old_weights, new_weights):
        """
        Calculates the local optimum for the given old and new weights.
        Equivalent to equation 19in the Abstract from Dr. Jack Kuipers.
        Args:
            old_weights (numpy.ndarray): The old weights.
            new_weights (numpy.ndarray): The new weights.

        Returns:
            numpy.ndarray: The local optimum.
        """
        a = self.order_weights * (self.cell_ratios - 1)
           
        b = 1 - old_weights * a + old_weights * (self.cell_ratios - 1)
        
        c = a / b
        
        def local_ll_sum(x, c):
            return -np.sum(np.log(new_weights * a / b + 1))
        
        res = minimize(local_ll_sum, x0=0.5, bounds=[(0, 1)], args=(c), method='L-BFGS-B', options={'ftol': 0.01})

        if not res.success:
            print(f"Minimization not successful, Reason: {res.message}")
        return res.x
    
    
    def get_optimal_weights(self, abs_diff=0.1, max_iter = 1000):
        """
            Calculates the optimal weights for the NEM model using the specified relative error and maximum number of iterations.

            Args:
            - abs_diff: a float representing the absolute difference threshold for convergence (default: 0.1)
            - max_iter: an integer representing the maximum number of iterations (default: 1000)

            Returns:
            - parent_weights: a list of length num_s containing the optimal weights for each variable's parents
            
            TODO:
            - Check if log-likelihood ratios are updated accordingly
            - Check if function is too flat (if not, one could decrease abs_diff, need to tackle overflows)
        """
        old_ll = -float('inf')
        ll_diff = float('inf')
        self.compute_ll_ratios()
        ll = np.log(self.compute_likelihood_score())
        iter_count = 0
        print(f"Initial LL: {ll}")
        while(ll_diff > abs_diff and iter_count < max_iter):
            self.calculate_order_weights()
            old_weights = self.parent_weights
            # new_weights = self.parent_weights
            
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    self.parent_weights[i][j] = self.calculate_local_optimum(old_weights[i][j], self.parent_weights[i][j])
            
            self.compute_ll_ratios()
            old_weights = self.parent_weights
            ll = np.log(self.compute_likelihood_score())
            ll_diff = ll - old_ll
            old_ll = ll
            iter_count += 1
            
            print(f"Iteration {iter_count}: LL: {ll}")
        
        
        