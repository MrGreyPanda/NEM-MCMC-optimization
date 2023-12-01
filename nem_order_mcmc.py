import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
import random
import utils
import copy
from itertools import cycle
from multiprocessing import Pool
import wandb
from scipy.linalg import solve_triangular


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
        self.U = nem.U.copy()
        self.perm_orders = []
        self.perm_orders.append(perm_order)
        self.parent_weights = np.zeros((self.num_s, self.num_s))
        self.score_tables = nem.get_score_tables(nem.observed_knockdown_mat)
        self.get_permissible_parents(perm_order, init=True)
        self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.score_tables)
        self.perm_order = perm_order

    def reset(self, perm_order, i1=None, i2=None, init=False):
        """
        Resets the instance of the NEMOrderMCMC class.

        Args:
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.
        """
        self.ll = 0.0
        self.get_permissible_parents(perm_order, i1, i2, init=init)

    def get_permissible_parents(self, perm_order, i1=None, i2=None, init=False):
        """
        Initializes the permissible parents and their weights for each node in the network given a permutation order.

        Args:
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.

        """
        parents_list = np.empty(self.num_s, dtype=object)
        n_parents = np.empty(self.num_s, dtype=int)
        if not init:
            self.parent_weights[i1] = 0
            self.parent_weights[i2] = 0
            self.parent_weights[:, i1] = 0
            self.parent_weights[:, i2] = 0
        for i in range(self.num_s):
            index = np.where(perm_order == i)[0]
            if len(index) > 0:
                index = index[0]
                parents_list[i] = perm_order[:index]
                n_parents[i] = len(parents_list[i])
                if init:
                    for j in parents_list[i]:
                        self.parent_weights[i][j] = 0.5
                else:
                    if i1 in parents_list[i]:
                        self.parent_weights[i][i1] = 0.5
                    elif i2 in parents_list[i]:
                        self.parent_weights[i][i2] = 0.5
                    elif i == i1 or i == i2:
                        for j in parents_list[i]:
                            self.parent_weights[j][i] = 0.5

            else:
                parents_list[i] = np.array([])
                n_parents[i] = 0

        
        self.parents_list, self.n_parents = parents_list, n_parents

    def compute_ll_ratios(self, weights, score_tables):
        """
        Computes the log-likelihood ratios for each cell in the NEM matrix.
        Equivalent to Equation 13 in the Abstract from Dr. Jack Kuipers.

        Returns:
        - numpy.ndarray: The log-likelihood ratios for each cell in the NEM matrix.
        """
        cell_ratios = self.U.copy()
        for i in range(self.num_s): 
            for j in self.parents_list[i]:
                cell_ratios[i, :] += np.log(1.0 -
                                            weights[i][j] +
                                            weights[i][j] *
                                            np.exp(score_tables[i][j]))
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
    
    def optimize_ll(self, W):
        W_matrix = W.reshape((self.num_s, self.num_s))
        cell_ratios = self.U.copy()
        for i in range(self.num_s):
            for j in self.parents_list[i]:
                cell_ratios[i, :] += np.log(1.0 -
                                            W_matrix[i][j] +
                                            W_matrix[i][j] *
                                            np.exp(self.score_tables[i][j]))
        cell_sums = np.logaddexp.reduce(cell_ratios, axis=0)
        ll = sum(cell_sums) 
        return -ll


    def calculate_local_optimum(self, i, j):
        """
        Calculates the local optimum for the given old and new weights.
        Equivalent to equation 19in the Abstract from Dr. Jack Kuipers.
        Args:
            weights:

        Returns:
            numpy.ndarray: The local optimum.
        """
        local_vec = np.exp(self.score_tables[i][j])
        a = (local_vec - 1.0) * self.order_weights[i]
        b = 1.0 - self.parent_weights[i][j] * a + self.parent_weights[i][j] * (local_vec - 1.0)
        c = a / b

        res = minimize(local_ll_sum, x0=0.5, bounds=[(0.0, 1.0)], args=(c,), method='L-BFGS-B', tol=0.1)
        if res.success is False:
            raise Exception(f"Minimization not successful, Reason: {res.message}")
        return res.x

    
    def calculate_optimum_greedy(self, i, j):
        local_vec = np.exp(self.score_tables[i][j]) - 1.0
        a = local_vec * self.order_weights[i]
        b = 1.0 - self.parent_weights[i][j] * a + self.parent_weights[i][j] * local_vec
        c = a / b
        grad = -np.sum(c / (c * self.parent_weights[i][j] + 1.0))
        
        return 0 if grad > 0 else 1
    
    def get_optimal_weights_greedy(self, abs_diff=0.01, max_iter=100, use_dag=True, i1=None, i2=None, init=False):
        old_ll = -float('inf')
        ll_diff = float('inf')
        iter_count = 0
        self.ll = 0.0
        # abs_diff could be varied
        while ll_diff > abs_diff and iter_count < max_iter:
            self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.score_tables)
            self.order_weights, self.ll = self.calculate_ll()
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                for j in self.parents_list[i]:
                    if i1 == j or i2 == j or i == i1 or i == i2:
                        new_parent_weights[i][j] = self.calculate_optimum_greedy(i, j)
            self.parent_weights = new_parent_weights
            ll_diff = self.ll - old_ll
            old_ll = self.ll
            iter_count += 1

        _, ll = self.calculate_ll()
        curr_ll = -float('inf')
        run = True
        iter_count = 0
        while(run and iter_count < 100):
            run = False
            new_parent_weights = self.parent_weights
            for i in range(self.num_s):
                    for j in self.parents_list[i]:
                        if i1 == j or i2 == j:
                            new_parent_weights[i][j] = 1 if self.parent_weights[i][j] == 0 else 0
                            _, curr_ll = self.calculate_ll()
                            if curr_ll > ll:
                                self.parent_weights = new_parent_weights
                                ll = curr_ll
                                run = True
            
            if run:
                self.parent_weights = new_parent_weights
            iter_count += 1
        return ll
    

    def get_optimal_weights(self, abs_diff=0.1, max_iter=40, use_nem=False, i1=None, i2=None, init=False, ultra_verbose=False):
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
        bounds = [(0.0, 1.0)] * self.num_s * self.num_s
        # abs_diff could be varied
        while ll_diff > abs_diff and iter_count < max_iter:
            self.cell_ratios = self.compute_ll_ratios(self.parent_weights, self.score_tables)
            self.order_weights, self.ll = self.calculate_ll()
            new_parent_weights = self.parent_weights.copy()
            if init:
                # for i in range(self.num_s):
                #     for j in self.parents_list[i]:
                #         new_parent_weights[i][j] = self.calculate_local_optimum(i, j)
                if ultra_verbose:
                    print(f"Iteration of weight optimization: {iter_count}")
                new_parent_weights = minimize(self.optimize_ll, x0=new_parent_weights.flatten(), method='L-BFGS-B', tol=0.1, bounds=bounds).x.reshape((self.num_s, self.num_s))
            else:
                for i in range(self.num_s):
                    for j in self.parents_list[i]:
                        if i1 == j or i2 == j or i == i1 or i == i2:
                            new_parent_weights[i][j] = self.calculate_local_optimum(i, j)
            self.parent_weights = new_parent_weights
            ll_diff = self.ll - old_ll
            old_ll = self.ll
            print(f"LL: {self.ll}")
            iter_count += 1
        if use_nem:
            _, dag_weights = self.create_nem(self.parent_weights)
        else:
            _, dag_weights = self.create_dag(self.parent_weights)
        dag_ll = utils.compute_ll(self.compute_ll_ratios(dag_weights, self.score_tables))
        return dag_ll
    
    def create_dag(self, weights):
        dag_weights = weights.copy()
        dag_weights = 1 * (weights > 0.5)
        dag = dag_weights.T     # transpose to get the correct dag (Doing it that way because of access efficiency)
        return dag, dag_weights
    
    def create_nem(self, weights):
        nem_weights = weights.copy()
        nem_weights = 1 * (weights > 0.5)
        nem_weights = utils.ancestor(nem_weights)
        nem = nem_weights.T
        return nem, nem_weights

        
    def accepting(self, score, curr_score, gamma, net, curr_net, perm_order, curr_perm_order):
        acceptance_rate = np.exp(gamma * (score - curr_score))
        if random.random() < acceptance_rate:
            return True, score, net, perm_order
        else:
            return False, curr_score, curr_net, curr_perm_order
        
    def get_new_order(self, curr_perm_order, swap_prob=0.95):
        """
        Swaps two adjacent nodes in the permutation order with a probability of swap_prob.

        Args:
        - swap_prob (float): Probability of swapping two nodes in the permutation order.

        Returns:
        - perm_order (numpy.ndarray): A 1D numpy array representing the permutation order of the nodes in the network.
        """
        # counter = 0
        perm_order = curr_perm_order.copy()
        
        # while np.any(np.all(perm_order == self.perm_orders, axis=1)):
        # if counter < self.num_s * 5:
        #     perm_order = curr_perm_order.copy()
        is_swap = random.random() < swap_prob
        if is_swap:
            # swap two random nodes
            i, j = random.sample(range(self.num_s), 2)
            i1 = np.where(perm_order == i)[0][0]
            i2 = np.where(perm_order == j)[0][0]
            perm_order[i], perm_order[j] = perm_order[j], perm_order[i]
        else:
            # swap two adjacent nodes
            i = random.randint(0, self.num_s - 2)
            j = i + 1
            i1 = np.where(perm_order == i)[0][0]
            i2 = np.where(perm_order == j)[0][0]
            perm_order[i], perm_order[j] = perm_order[j], perm_order[i]
            # counter += 1
        # wandb.log({"perm_order_counter": counter})
        # self.perm_orders.append(perm_order)
        return perm_order, i1, i2
   
    def method(self, swap_prob=0.95, gamma=1, seed=1234, n_iterations=500, verbose=True, ultra_verbose=False, use_nem=False):
        """
        Runs the MCMC algorithm to find the optimal permutation order for the NEM model.

        Args:
        - swap_prob (float): Probability of swapping two nodes in the permutation order.
        - gamma (float): Scaling factor for the acceptance rate calculation.
        - seed (int): Seed for the random number generator.
        - n_iterations (int): Number of iterations to run the MCMC algorithm.

        Returns:
        - best_score (float): The highest score achieved during the MCMC iterations.
        - best_nem (list): The optimal NEM model found during the MCMC iterations.
        """
        curr_score = self.get_optimal_weights(init=True, ultra_verbose=ultra_verbose, use_nem=use_nem)
        best_score = curr_score
        dag, _ = self.create_dag(self.parent_weights)
        best_dag = dag
        curr_perm_order = self.perm_order
        perm_order = curr_perm_order
        best_order = perm_order
        best_order_list = [best_order]
        curr_dag = np.zeros((self.num_s, self.num_s))
        curr_score_list = [curr_score]
        best_score_list = [best_score]
        all_score_list = [curr_score]
        best_parents_list = self.parents_list.copy()
        for i in range(n_iterations):
            if verbose and i % 50 == 0:
                print(f"{i}-th iteration")
            perm_order, i1, i2 = self.get_new_order(curr_perm_order, swap_prob=swap_prob)
            self.reset(perm_order=perm_order, i1=i1, i2=i2)
            ll = self.get_optimal_weights(init=True, use_nem=use_nem, ultra_verbose=ultra_verbose)
            wandb.log({"ll-score": ll})
            # self.reset(perm_order=perm_order, i1=i1, i2=i2, init=False)
            # ll = self.get_optimal_weights_greedy(i1=i1, i2=i2)
            all_score_list.append(ll)
            if use_nem:
                dag, _ = self.create_nem(self.parent_weights)
            else:
                dag, _ = self.create_dag(self.parent_weights)
            curr_score_list.append(curr_score)
            acc, curr_score, curr_dag, curr_perm_order = self.accepting(ll, curr_score, gamma, dag, curr_dag, perm_order, curr_perm_order)
            perm_order = curr_perm_order
            if acc:
                wandb.log({"curr_score": curr_score})
                wandb.log({"Current Score - Real Score": self.nem.obs_ll - curr_score})
                if curr_score > best_score:
                    best_score = curr_score
                    best_dag = dag
                    wandb.log({"Hamming Distance": utils.hamming_distance(best_dag, self.nem.adj_matrix)})
                    best_order = curr_perm_order.copy()
                    best_parents_list = self.parents_list.copy()
                    best_score_list.append(best_score)
                    best_order_list.append(best_order)
                    wandb.log({"best_score": best_score})
                    wandb.log({"Best Score - Real Score": self.nem.obs_ll - best_score})
        self.best_score = best_score
        self.best_dag = best_dag
        self.best_order = best_order
        self.all_score_list = all_score_list
        self.curr_score_list = curr_score_list
        self.best_score_list = best_score_list
        self.parents_list = best_parents_list
        return best_score, best_dag
    
    def condition(self, i, j):
        return i in self.parents_list[j]


def replica_exchange_step(replicas, gammas, n_replicas, n_iters, scores, upwards_cylce):
    """
    Runs the replica exchange MCMC algorithm to find the optimal permutation order for the NEM model.

    Args:
    - gammas (list): List of scaling factors for the acceptance rate calculation for each replica.
    - n_replicas (int): Number of replicas to run the replica exchange MCMC algorithm.
    - n_steps (int): Number of steps to run the replica exchange MCMC algorithm.

    Returns:
    - best_score (float): The highest score achieved during the MCMC iterations.
    - best_nem (NEMOrderMCMC): The optimal NEM model found during the MCMC iterations.
    """
    n_exchanges = 0
    for i in range(n_replicas):
        replicas[i].method(n_iterations=n_iters, gamma=gammas[i], verbose=True)
        replicas[i].perm_orders = [replicas[i].perm_orders[-1]]
        scores[i] = replicas[i].best_score
    best_score = np.max(scores)
    print(f"Best scoring nem {np.argmax(scores)}")
    best_nem = replicas[np.argmax(scores)]

    if upwards_cylce:
        # swap (0,1), (2,3), ...
        partners = [(j-1, j) for j in range(1, n_replicas, 2)]
    else:
        # swap (1,2), (3,4), ...
        partners = [(j-1, j) for j in range(2, n_replicas, 2)]
    for (i,j) in partners:
        delta = gammas[i] * scores[j] - gammas[i] * scores[i] + gammas[j] * scores[i] - gammas[j] * scores[j]
        if random.random() < np.exp(-delta):
            replicas[i], replicas[j] = replicas[j], replicas[i]
            scores[i], scores[j] = scores[j], scores[i]
            n_exchanges += 1
            if scores[i] > best_score:
                best_score = scores[i]
                best_nem = replicas[i]
    # Need to look at edge cases
    return best_score, best_nem, replicas, scores, n_exchanges

def replica_exchange_method(nem, n_exchange, n_iter, init_order_guess):
    n_replicas = 10
    perm_order = init_order_guess
    gammas = []
    replicas = []
    n_replicas = 10
    for i in range(n_replicas):
        gammas.append((1.0 + i * 0.2) * nem.num_s / nem.num_e)
        replicas.append(NEMOrderMCMC(nem, perm_order))
    scores = np.zeros(n_replicas)
    n_exchanges = 0
    cycler = cycle([True, False])
    for i in range(n_exchange):
        print(f"i-th exchanges: {i}")
        best_score, best_nem, replicas, scores, exchanges = replica_exchange_step(replicas, gammas, n_replicas, n_iter, scores, next(cycler))
        n_exchanges += exchanges
        print(f"Best score: {best_score}")
    print(f"Number of exchanges: {n_exchanges}")
    print(f"Best DAG: {best_nem.best_dag}")
    return best_score, best_nem




