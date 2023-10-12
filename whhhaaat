import numpy as np
import utils
import os

class NEM:
    """
    A class representing a Network Error Model (NEM).

    Attributes:
    - num_s (int): the number of s-points in the network
    - num_e (int): the number of e-points in the network
    - s_mat (numpy.ndarray): the adjacency matrix of the network
    - e_arr (numpy.ndarray): the end nodes of the network
    - A (float): the value of the parameter A in the NEM
    - B (float): the value of the parameter B in the NEM
    - observed_knockdown_mat (numpy.ndarray): the connection matrix of the network

    Methods:
    - __init__(self): initializes the NEM object by reading the network.csv file and setting the attributes
    """

    def __init__(self):
        """
        Initializes the NEM object by reading the network.csv file and setting the attributes.
        """
        
        # Get the current directory
        current_dir = os.getcwd()

        # Read the network.csv file
        with open(os.path.join(current_dir, 'network.csv'), 'r') as f:
            # Read the first line to get num_s and num_e
            # Read the first line to get num_s and num_e
            self.num_s, self.num_e = map(int, f.readline().strip().split(','))
            # Create an empty adjacency matrix
            adj_matrix = np.zeros((self.num_s, self.num_s), dtype=int)

            # Read the connected s-points and update the adjacency matrix
            for line in f:
                s_points = [map(int, line.strip().split(','))]
                if len(s_points) != 2:
                    break
                adj_matrix[s_points[0]-1, s_points[1]-1] = 1
                
            # Read the last line to get the end nodes
            self.e_arr = np.array(line.strip().split(',')).astype(int)
            alpha, beta = map(float, f.readline().strip().split(','))
            

        # Save the adjacency matrix and end nodes as class members
        self.s_mat = adj_matrix
        print(len(self.e_arr))
        self.A = np.log(alpha / (1.0 - beta))
        self.B = np.log(beta / (1.0 - alpha))
        self.real_knockdown_mat = utils.create_real_knockdown_mat(self.s_mat, self.e_arr)
        self.observed_knockdown_mat = utils.create_observed_knockdown_mat(self.real_knockdown_mat, alpha, beta)
        self.all_score_tables = self.get_score_tables()
        self.U = self.get_node_lr_table()
        self.parent_weights = [[]]
        # self.parentLists = utils.create_parent_lists(self.s_mat)
        self.parentsLists = [[]]
        for index, list in enumerate(self.parentsLists):
            for i in list:
                self.weights[index].append(0.5)
                
        self.reduced_score_tables = self.get_reduced_score_tables(self.parentsLists)
        self.cellLRs = self.compute_ll_ratios()
        
        
    def compute_scores(self, node):
        """
        Computes the scores for a given set of effect nodes, data, and parameters A and B.
        Returns:
        numpy.ndarray: The computed scores.
        """
        # suppose only effect_nodes have an effect on E-genes upon perturbation i.e. we expect to see all 1
        print("shape knockdown_mat", self.observed_knockdown_mat.shape)
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
        

    def get_node_lr_table(self):
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
            l.append(self.all_score_tables[i][i]) # return base row

        res = np.vstack(l)
        res = np.vstack((res, np.where(self.observed_knockdown_mat == 0, 0, self.A).sum(axis=0)))

        return res
    
    def get_reduced_score_tables(self, parentLists):
        """
        Returns a list of reduced score tables based on the given parent lists.

        Args:
        parentLists (list): A list of lists containing the indices of the parents for each student.

        Returns:
        list: A list of numpy arrays containing the scores of each student's parents.
        """
        reduced_score_tables = [
            np.array([self.all_score_tables[i][j] for j in parentLists[i]])
            for i in range(self.num_s)
        ]
        return reduced_score_tables
    
    def compute_ll_ratios(self):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.

        Returns:
        cellLRs (numpy.ndarray): A 2D numpy array of shape (num_s, num_t) containing the log-likelihood ratios for each cell in the NEM matrix.
        """
        nparents = [len(self.parentsLists[i]) for i in range(self.num_s)]
        cellLRs = self.U
        for ii in range(self.num_s):
            if nparents[ii] > 1:
                cellLRs[ii, :] = cellLRs[ii, :] + np.sum(np.log(1 - self.parent_weights[ii] + self.parent_weights[ii] * np.exp(self.reduced_score_tables[ii])), axis=0)
            if nparents[ii] == 1:
                cellLRs[ii, :] = cellLRs[ii, :] + np.log(1 - self.parent_weights[ii] + self.parent_weights[ii] * np.exp(self.reduced_score_tables[ii]))
        return cellLRs
            
#     # loop over S-genes
#   # for each subsetted score table: sum over rows(S-genes) to obtain a vector of dim 1 x #E-genes
#   for (ii in 1:n) { 
#     if (nparents[ii] > 1) {
#       cellLRs[ii, ] <- as.numeric(cellLRs[ii, ] + colSums(log(1 - parentWeights[[ii]] + parentWeights[[ii]]*exp(parentRs[[ii]]))))
#     }
#     if (nparents[ii] == 1) {
#       cellLRs[ii, ] <- as.numeric(cellLRs[ii, ] + log(1 - parentWeights[[ii]] + parentWeights[[ii]]*exp(parentRs[[ii]])))
#     }
#   }
#   cellLRs
# 
# loop over S-genes
# for each subsetted score table: sum over rows(S-genes) to obtain a vector of dim 1 x #E-genes
