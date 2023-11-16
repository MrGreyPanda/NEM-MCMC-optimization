import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
import random
import utils


def local_ll_sum(x, c):
    res = -np.sum(np.log(c * x + 1.0))
    return res


class NEMOrderMCMC:
    def __init__(self, nem, perm_order):
        """
        Initializes an instance of the NEMOrderMCMC class.

        Args:
        - nem (NEM): an instance of the NEM class
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.
        """
        self.nem = nem
        self.num_s = nem.num_s
        self.U = nem.U
        self.score_table_list = nem.get_score_tables(nem.observed_knockdown_mat)
        self.get_permissible_parents(perm_order)
        self.reduced_score_tables = self.get_reduced_score_tables(self.score_table_list, self.parents_list)
        self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.reduced_score_tables)
        self.perm_order = perm_order

    def reset(self, perm_order):
        """
        Resets the instance of the NEMOrderMCMC class.

        Args:
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.
        """
        self.ll = 0.0
        self.get_permissible_parents(perm_order)
        self.reduced_score_tables.clear()
        self.reduced_score_tables = self.get_reduced_score_tables(self.score_table_list, self.parents_list)

    def get_permissible_parents(self, perm_order):
        """
        Initializes the permissible parents and their weights for each node in the network given a permutation order.

        Args:
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.

        """
        parent_weights = np.empty(self.num_s, dtype=object)
        parents_list = np.empty(self.num_s, dtype=object)
        n_parents = np.empty(self.num_s, dtype=int)
        for i in range(self.num_s):
            index = np.where(perm_order == i)[0][0]
            parents_list[i] = perm_order[index + 1:]
            n_parents[i] = len(parents_list[i])
            parent_weights[i] = [0.5] * n_parents[i]
        self.parents_list, self.n_parents, self.parent_weights = parents_list, n_parents, parent_weights

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

    def compute_ll_ratios(self, weights, reduced_score_tables):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.

        Returns:
        - numpy.ndarray: The log-likelihood ratios for each cell in the NEM matrix.
        """
        cell_ratios = self.U.copy()
        for i in range(self.num_s): 
            for j in range(self.n_parents[i]):
                cell_ratios[i, :] += np.log(1.0 -
                                            weights[i][j] +
                                            weights[i][j] *
                                            np.exp(reduced_score_tables[i][j]))
        return cell_ratios

    def calculate_ll(self):
        """
        Calculates the log-likelihood of the NEM model.

        Returns:
        - tuple: A tuple containing the order weights and the log-likelihood of the NEM model.
        """
        cell_sums = np.logaddexp.reduce(self.cell_ratios, axis=0)
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

        res = minimize(local_ll_sum, x0=0.5, bounds=[(0.0, 1.0)], args=(c,), method='L-BFGS-B', tol=0.1)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x
    
    def calculate_optimum_greedy(self, i, j):
        local_vec = np.exp(self.reduced_score_tables[i][j])
        a = (local_vec - 1.0) * self.order_weights[i]
        b = 1.0 - self.parent_weights[i][j] * a + self.parent_weights[i][j] * (local_vec - 1.0)
        c = a / b
        grad = -np.sum(c / (c * self.parent_weights[i][j] + 1.0))
        
        return 0 if grad > 0 else 1
    
    def get_optimal_weights_greedy(self, abs_diff=0.01, max_iter=100, use_dag=True):
        old_ll = -float('inf')
        ll_diff = float('inf')
        iter_count = 0
        self.ll = 0.0
        # abs_diff could be varied
        while ll_diff > abs_diff and iter_count < max_iter:
            self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.reduced_score_tables)
            self.order_weights, self.ll = self.calculate_ll()
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    new_parent_weights[i][j] = self.calculate_optimum_greedy(i, j)
            self.parent_weights = new_parent_weights
            ll_diff = self.ll - old_ll
            old_ll = self.ll
            iter_count += 1

        

    def get_optimal_weights(self, abs_diff=0.01, max_iter=100, use_dag=True):
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
        # abs_diff could be varied
        while ll_diff > abs_diff and iter_count < max_iter:
            self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.reduced_score_tables)
            self.order_weights, self.ll = self.calculate_ll()
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    new_parent_weights[i][j] = self.calculate_optimum_greedy(i, j)
            self.parent_weights = new_parent_weights
            ll_diff = self.ll - old_ll
            old_ll = self.ll
            iter_count += 1

        print(f"Number of iterations: {iter_count}")
        dag_weights = self.parent_weights
        dag = np.zeros((self.num_s, self.num_s))
        for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    dag_weights[i][j] = 1 * (self.parent_weights[i][j] > 0.5)
                    dag[self.parents_list[i][j], i] = dag_weights[i][j]
        
        dag_ll = utils.compute_ll(self.compute_ll_ratios(dag_weights, self.reduced_score_tables))
        return dag_ll
    
        
    def accepting(self, score, curr_score, gamma, dag, perm_order, curr_perm_order):
        acceptance_rate = np.exp(gamma * (score - curr_score))
        dag_weights = self.parent_weights
        if random.random() < acceptance_rate:
            dag = np.zeros((self.num_s, self.num_s))
            
            for i in range(self.num_s):
                for j in range(self.n_parents[i]):
                    dag_weights[i][j] = 1 * (self.parent_weights[i][j] > 0.5)
                    dag[self.parents_list[i][j], i] = dag_weights[i][j]
            return score, dag, perm_order
        else:
            return curr_score, dag, curr_perm_order

        

    def method(self, swap_prob=0.95, gamma=1, seed=1234, n_iterations=500, is_dag=True):
        """
        Runs the MCMC algorithm to find the optimal permutation order for the NEM model.

        Args:
        - swap_prob (float): Probability of swapping two nodes in the permutation order.
        - gamma (float): Scaling factor for the acceptance rate calculation.
        - seed (int): Seed for the random number generator.
        - n_iterations (int): Number of iterations to run the MCMC algorithm.
        - is_dag (bool): Whether the graph is a directed acyclic graph (DAG).

        Returns:
        - best_score (float): The highest score achieved during the MCMC iterations.
        - best_nem (list): The optimal NEM model found during the MCMC iterations.
        """
        random.seed(seed)
        curr_perm_order = self.perm_order
        curr_score = self.get_optimal_weights()
        best_score = curr_score
        perm_order = curr_perm_order
        best_dag = np.zeros((self.num_s, self.num_s))
        
        dag = np.zeros((self.num_s, self.num_s))
        for i in range(n_iterations):
            print(f"Starting MCMC iteration {i}")
            is_swap = random.random() < swap_prob

            if is_swap:
                # swap two random nodes
                i, j = random.sample(range(self.num_s), 2)
                perm_order[i], perm_order[j] = curr_perm_order[j], curr_perm_order[i]
            else:
                # swap two adjacent nodes
                i = random.randint(0, self.num_s - 2)
                perm_order[i], perm_order[i + 1] = curr_perm_order[i + 1], curr_perm_order[i]
            self.reset(perm_order=perm_order)
            self.get_optimal_weights()
            curr_score, dag, curr_perm_order= self.accepting(self.ll, curr_score, gamma, dag, perm_order, curr_perm_order)
            print(f"Score: {self.ll}, Current Score: {curr_score}")
            
            if curr_score > best_score:
                best_score = curr_score
                best_dag = dag
            
        return best_score, best_dag
