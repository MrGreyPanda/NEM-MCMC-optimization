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
        self.U = self.get_node_lr_table(self.get_score_tables())
        
        
        
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
    
    def build_score_table(self, node):
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
        score_table[node, :] = self.compute_scores(node)
        
        # compute the increment of score if we have additional parents
        indices = {i for i in range(self.num_s) if i != node}
        for index in indices:
            score_table[index, :] = np.where(self.observed_knockdown_mat[index] == 0, self.B, -self.A)
        
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
        
        Returns:
        numpy.ndarray: A table with the node scores for all effect nodes in the network.
        """
        l = []
        for i in range(self.num_s):
            l.append(all_score_tables[i][i]) # return base row
            
        res = np.vstack(l)
        res = np.vstack((res, np.where(self.observed_knockdown_mat == 0, 0, self.A).sum(axis=0)))
        
        return res
    
    