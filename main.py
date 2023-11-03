import nem
from nem_order_mcmc import NEMOrderMCMC
import utils
import numpy as np
from DAGs import dot
from DAGs import graph
import os

def get_initial_order(observed_knockdown_mat):
    """
    Make an "educated" guess on the order of the nodes in the network.
    """
    num_s = observed_knockdown_mat.shape[0]
    sum_col = np.sum(observed_knockdown_mat, axis=1)
    order = np.arange(num_s)
    print(order)
    order = np.argsort(-sum_col)
    print(sum_col)
    print(order)
    return order

def main():
    # logging.basicConfig(filename='nem_mcmc.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj("DAGs/csv/network10.csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    # permutation_order = np.array(random.sample(range(num_s), num_s))
    permutation_order = get_initial_order(my_nem.observed_knockdown_mat)
    gamma = 2.0 * my_nem.num_s / my_nem.num_e
    mcmc_nem = NEMOrderMCMC(my_nem, permutation_order)
    
    score, best_nem = mcmc_nem.method(n_iterations=100, gamma=gamma)
    # for i in range(my_nem.num_s):
    #     best_nem = utils.ancestor(best_nem)
    if os.path.exists("output.pdf"):
        os.remove("output.pdf")
    print(f"Final Score: {score}")
    dot.generate_dot_from_matrix(best_nem, "output.dot")
    graph.create_graph_from_dot("output.dot", "output.pdf")
    
if __name__ == "__main__":
    main()