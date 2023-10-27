import nem
# from custom_nem import CustomNEMOptimizer
from nem_order_mcmc import NEMOrderMCMC
import utils
import numpy as np
import random
import csv

def main():
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj("DAGs/csv/network3.csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    # custom = CustomNEMOptimizer(my_nem)
    # nem.get_optimal_weights()
    print(f"A: {my_nem.A}, B: {my_nem.B}")
    permutation_order = np.array(random.sample(range(num_s), num_s))
    mcmc_nem = NEMOrderMCMC(my_nem, permutation_order)
    
    result = mcmc_nem.method(n_iterations=50)
    permutation = result[2][result[0].index(max(result[0]))]
    print(permutation)
    
if __name__ == "__main__":
    main()