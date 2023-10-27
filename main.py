import nem
# from custom_nem import CustomNEMOptimizer
from nem_order_mcmc import NEMOrderMCMC
import utils
import numpy as np
import random
import csv

def main():
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj("network.csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    # custom = CustomNEMOptimizer(my_nem)
    # nem.get_optimal_weights()
    permutation_order = np.array(random.sample(range(num_s), num_s))
    mcmc_nem = NEMOrderMCMC(my_nem, permutation_order)
    mcmc_nem.get_optimal_weights()
    
    result = mcmc_nem.method()
    
    with open('result.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['List 1', 'List 2', 'List 3'])
            for i in range(len(result[0])):
                writer.writerow([result[0][i], result[1][i], result[2][i]])
    
    

if __name__ == "__main__":
    main()