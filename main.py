import nem
from custom_nem import CustomNEMOptimizer

def main():
    my_nem = nem.NEM("network.csv")
    custom = CustomNEMOptimizer(my_nem)
    
    
    

if __name__ == "__main__":
    main()