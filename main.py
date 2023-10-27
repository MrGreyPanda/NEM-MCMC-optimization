import nem
from custom_nem import CustomNEMOptimizer
import utils

def main():
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj("network.csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    # custom = CustomNEMOptimizer(my_nem)
    my_nem.get_optimal_weights()
    
    
    

if __name__ == "__main__":
    main()