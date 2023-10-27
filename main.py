import nem
from custom_nem import CustomNEMOptimizer

def main():
    my_nem = nem.NEM("network.csv")
    # custom = CustomNEMOptimizer(my_nem)
    my_nem.get_optimal_weights()
    
    
    

if __name__ == "__main__":
    main()