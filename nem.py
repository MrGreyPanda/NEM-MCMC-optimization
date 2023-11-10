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
        
        self.A = np.log(alpha / (1.0 - beta))
        self.B = np.log(beta / (1.0 - alpha))
        self.observed_knockdown_mat = utils.create_observed_knockdown_mat(real_knockdown_mat, alpha, beta)
        self.U = self.get_node_lr_table(self.get_score_tables(self.observed_knockdown_mat))
        self.real_ll = self.compute_real_score(real_knockdown_mat=real_knockdown_mat)

    def compute_scores(self, node, knockdown_mat):
        """
        Computes the scores for a given set of effect nodes, data, and parameters A and B.
        Returns:
        numpy.ndarray: The computed scores.
        """
        # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
        score = np.where(knockdown_mat[node, :] == 1, 0, self.B)  # real 1, observe 1 -> 0. real 1 observe 0, FN-> B

        # suppose the rest does not have an effect on E-genes, therefore if we perturb them we expect to see 0
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            score += np.where(knockdown_mat[index, :] == 1, self.A, 0)  # real 0, observe 0 -> 0. real 0 observe 1, FP-> A

        return score
    
    def build_score_table(self, node, knockdown_mat):
        """
        Builds a score table for a given node in a network.

        Parameters:
        node (int): The node for which to build the score table.
 
        Returns:
        pandas.DataFrame: The score table for the given node.
        """
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
            all_score_tables.append(self.build_score_table(node, knockdown_mat))
        return all_score_tables

    def get_node_lr_table(self, all_score_tables):
        """
        Returns a table with the node scores for all effect nodes in the network.
        
        Parameters:
        all_score_tables (list): A list containing score tables for all genes in the network.
        
        Returns:
        numpy.ndarray: A table with the node scores for all effect nodes in the network.
        """
        l = []
        for i in range(self.num_s):
            l.append(all_score_tables[i][i])  # return base row
            
        res = np.vstack(l)
        res = np.vstack((res, np.where(self.observed_knockdown_mat == 0, 0, self.A).sum(axis=0)))
        
        return res
    
    def get_reduced_score_tables(self, score_table_list, parents_list):
        """
        Initializes a list of reduced score tables based on the given score_table_list.

        Args:
        score_table_list (list(np.array)): a list containing the score tables for each gene in the network.
        """
        reduced_score_tables = []
        for i in range(self.num_s):
            reduced_score_tables.append(np.array([score_table_list[i][j] for j in parents_list[i]]))
        return reduced_score_tables
    
    def compute_ll_ratios(self, parent_weights, reduced_score_tables, real_n_parents):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.
       
        Returns:
        - numpy.ndarray: The log-likelihood ratios for each cell in the NEM matrix.
        """
        cell_ratios = self.U.copy()
        for i in range(self.num_s):        # iterate through all nodes
            for j in range(real_n_parents[i]):
                cell_ratios[i, :] += np.log(1.0 - 
                                            parent_weights[i][j] + 
                                            parent_weights[i][j] * 
                                            np.exp(reduced_score_tables[i][j]))
        return cell_ratios
    
    def calculate_ll(self, cell_ratios):
        """
        Calculates the log-likelihood of the NEM model.

        Returns:
        - tuple: A tuple containing the order weights and the log-likelihood of the NEM model.
        """
        max_val = np.max(cell_ratios, axis=0)
        cell_sums = np.log(np.sum(np.exp(cell_ratios - max_val), axis=0)) + max_val
        ll = sum(cell_sums)
        return ll
    
    def compute_real_score(self, real_knockdown_mat):
        """
        Computes the log-likelihood score of the given real knockdown matrix.

        Args:
            real_knockdown_mat (numpy.ndarray): A binary matrix of shape (num_s, num_p) where num_s is the number of
                samples and num_p is the number of perturbations. A value of 1 in position (i, j) indicates that sample i
                was knocked down for perturbation j.

        Returns:
            None. The method updates the `real_ll` attribute of the object with the computed log-likelihood score.
        """
        row_sums = np.sum(real_knockdown_mat, axis=1)
        sorted_indices = np.argsort(row_sums)[::-1]
        real_parent_order = np.arange(self.num_s)[sorted_indices]
        real_score_table_list = self.get_score_tables(real_knockdown_mat)
        real_n_parents = np.empty(self.num_s, dtype=int)
        real_parents_list = np.empty(self.num_s, dtype=object)
        real_parent_weights = np.empty(self.num_s, dtype=object)
        for index in range(self.num_s):
            real_parents_list[index] = real_parent_order[index + 1:]
            real_n_parents[index] = len(real_parents_list[index])
            real_parent_weights[index] = []
            for j in range(real_n_parents[index]):
                real_parent_weights[index].append(real_knockdown_mat[index, real_parents_list[index][j]])
        real_reduced_score_tables = self.get_reduced_score_tables(real_score_table_list, real_parents_list)
        real_cell_ratios = self.compute_ll_ratios(real_parent_weights, real_reduced_score_tables, real_n_parents)
        return self.calculate_ll(real_cell_ratios)