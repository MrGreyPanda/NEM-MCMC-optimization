import nem
# from custom_nem import CustomNEMOptimizer
from nem_order_mcmc import NEMOrderMCMC
import utils
import numpy as np
import random
import csv
import logging

def main():
    logging.basicConfig(filename='nem_mcmc.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj("network.csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    permutation_order = np.array(random.sample(range(num_s), num_s))
    mcmc_nem = NEMOrderMCMC(my_nem, permutation_order)
    
    score, best_nem = mcmc_nem.method()
    print(f"Score: {score}")
    print("NEM:")
    print(best_nem)
    
if __name__ == "__main__":
    main()