from nem import NEM
import utils
import numpy as np
from methods import InverseMethod, Method
import wandb
import os
import random as rnd
import time

def gen_thesis_data():
    rnd.seed(42)
    # sigma = [[[3, 4, 1, 5, 1, 2, 6, 4, 3, 5]]]
    adj_mat = [[0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 1, 1],
               [1, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0,],
               [0, 0, 0, 0, 0, 0]]
    
    num_s = len(adj_mat[0])
    α = 0.05
    β = 0.08
    errors = [α, β]
    # e_vec = [2, 3, 0, 4, 0, 1, 5, 3, 2, 4]
    e_vec = [2,3,0,4,0,1,5,3,2,4,1,2,3,4,5,0,0,1,3,4,5,1,2,4,5,0,1,2,3,4,5,1,2,0,0,0,1,2,3,4,5]
    num_e = len(e_vec)
    nem = NEM(adj_mat, e_vec, errors, num_s, num_e)
    init_guess = utils.initial_order_guess(nem.observed_knockdown_mat)
    inv_method = InverseMethod(init_guess, num_s, num_e, nem.U, nem.get_score_tables(nem.observed_knockdown_mat))
    weights, ll = inv_method.optimize()
    print(f"Comparison: real_ll: {nem.obs_ll} - inferred_ll: {ll}")
    print(f"Hamming Distance: {np.sum(np.abs(weights - adj_mat))}")
    print(f"weights:\n{weights}")
    # print(weights)

def conduct_var_e_genes_experiments_w_reinit(seeds=[42], network_nrs=[12], Methods=[InverseMethod]):
    wandb.login()
    for network_nr in network_nrs:
        curr_dir = os.getcwd()
        network_nr = 12
        print(f"Network {network_nr}:")
        network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
        adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
        num_e_max = num_s * 30
                   
        for Method in Methods:
            for seed in seeds:
                end_nodes = []
                for _ in range(num_s - 1):
                    end_nodes.append(rnd.randint(0, num_s-1))
                rnd.seed(seed)
                run = wandb.init(
                project="MCMC-NEM",
                name=f"Var E Genes, reinitializing weights, network {network_nr}, n_e_max: {num_e_max}, Method: {Method.__name__}, seed: {seed}",
                # Track hyperparameters and run metadata
                config={
                    "Experiment": "Var E Genes, reinitializing weights",
                    "seed": seed,
                    "num_s": num_s,
                    "num_e_max": num_e_max,
                    "adj_matrix": adj_matrix,
                    "errors": errors
                })
                for j in range(num_s, num_e_max, 1):
                    num_e = j
                    end_nodes.append(rnd.randint(0, num_s-1))
                    print(f"num_e: {num_e}")
                    curr_nem = NEM(adj_matrix, end_nodes, errors, num_s, num_e, seed=seed)
                    permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
     
                    # Set the project where this run will be logged
                    print(f"num_s: {num_s}")
                    start = time.time()
                    method = Method(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
                    weights, ll = method.optimize(use_wandb=True)
                    end = time.time()
                    comp = curr_nem.obs_ll - ll
                    hamming_distance = np.sum(np.abs(weights - adj_matrix))
                    
                    wandb.log({"Comparison": comp})
                    wandb.log({"Hamming Distance": hamming_distance})
                    wandb.log({"num_e": num_e})
                    wandb.log({"Time elapsed (s)": end-start})
                    print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll} = {comp}")
                    print(f"Hamming Distance: {hamming_distance}")
                    print(f"Time elapsed (s): {end-start}")
                    
                    # print(f"weights:\n{weights}")
                wandb.finish()
                print("#################")

def conduct_var_e_genes_experiments(seeds=[42], network_nrs=[12], Methods=[InverseMethod]):
    wandb.login()
    for network_nr in network_nrs:
        curr_dir = os.getcwd()
        print(f"Network {network_nr}:")
        network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
        adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
        
        num_e_max = num_s * 30
        for seed in seeds:
            rnd.seed(seed)
            # for i in range(19):
            end_nodes = []
            for _ in range(num_s - 1):
                    end_nodes.append(rnd.randint(0, num_s-1))
            
            weights = None
            print(f"num_s: {num_s}")
            for j in range(num_s, num_e_max, 1):
                num_e = j
                print(f"num_e: {num_e}")
                end_nodes.append(rnd.randint(0, num_s-1))
                
                curr_nem = NEM(adj_matrix, end_nodes, errors, num_s, num_e)
                permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
                for Method in Methods:
                    run = wandb.init(
                    # Set the project where this run will be logged
                    project="MCMC-NEM",
                    name=f"Var E Genes, keeping old weights, network {network_nr}, n_e: {num_e}, Method: {Method.type.__name__}",
                    # Track hyperparameters and run metadata
                    config={
                        "Experiment": "Var E Genes, keeping old weights",
                        "seed": seed,
                        "num_s": num_s,
                        "num_e_max": num_e_max,
                        "adj_matrix": adj_matrix,
                        "errors": errors
                    })
                    method = Method(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
                    weights, ll = method.optimize(weights=weights, use_wandb=True)
                    wandb.log({"Comparison": curr_nem.obs_ll - ll})
                    wandb.log({"Hamming Distance": np.sum(np.abs(weights - adj_matrix))})
                    wandb.log({"num_e": num_e})
                    print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll}")
                    print(f"Hamming Distance: {np.sum(np.abs(weights - adj_matrix))}")
                


def conduct_fixed_e_genes_experiments():
    wandb.login()
    seed = 42
    rnd.seed(seed)
    for i in range(20):
        print(f"Network {i}:")
        curr_dir = os.getcwd()
        network_nr = i
        network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
        adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
        curr_nem = NEM(adj_matrix, end_nodes, errors, num_s, num_e, seed=seed)
        permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
        
        run = wandb.init(
        # Set the project where this run will be logged
        project="MCMC-NEM",
        # Track hyperparameters and run metadata
        config={
            "seed": seed,
            "num_s": num_s,
            "num_e": num_e,
            "adj_matrix": adj_matrix,
            "errors": errors,
            "permutation_order": permutation_order
        })
        start = time.time()
        inv_method = InverseMethod(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
        weights, ll = inv_method.optimize()
        end = time.time()
        wandb.log({"Comparison": curr_nem.obs_ll - ll})
        wandb.log({"Hamming Distance": np.sum(np.abs(weights - adj_matrix))})
        wandb.log({"Time elapsed (s)": end-start})
        print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll}")
        print(f"Hamming Distance: {np.sum(np.abs(weights - adj_matrix))}")
        # print(f"weights:\n{weights}")


def conduct_one_big_run(seeds=[42], network_nrs=[12], Method=InverseMethod):
    for network_nr in network_nrs:
        curr_dir = os.getcwd()
        network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
        adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
        print("Adjacency Matrix of network ",network_nr, ":\n", adj_matrix)
        num_e = num_s * 10
        run = wandb.init(
            project="MCMC-NEM",
            name=f"Big run, {str(Method)}, network {network_nr}, n_e: {num_e}",
            config={
                "network_nr": network_nr,
                "num_s": num_s,
                "num_e": num_e,
                "adj_matrix": adj_matrix,
                "errors": errors,
            })
        for seed in seeds:
            wandb.log({"seed": seed})
            rnd.seed(seed)
            end_nodes = []
            for _ in range(num_e):
                    end_nodes.append(rnd.randint(0, num_s-1))
            print(f"end_nodes: {end_nodes}")
            curr_nem = NEM(adj_matrix=adj_matrix, end_nodes=end_nodes, errors=errors, num_s=num_s, num_e=num_e, seed=seed)
            permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
            print(f"permutation_order: {permutation_order}")
            start = time.time()
            method = Method(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
            weights, ll = method.optimize(use_wandb=True)
            end = time.time()
            print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll}")
            print(f"Hamming Distance: {np.sum(np.abs(weights - adj_matrix))}")
            wandb.log({"Comparison": curr_nem.obs_ll - ll})
            wandb.log({"Hamming Distance": np.sum(np.abs(weights - adj_matrix))})
            wandb.log({"Time elapsed (s)": end-start})
            print(f"weights:\n{weights}")
        wandb.finish()
        print("#################")

# Comment out and run with python experiments.py
conduct_var_e_genes_experiments_w_reinit(Methods=[Method])
# conduct_var_e_genes_experiments()
# conduct_fixed_e_genes_experiments()
# gen_thesis_data()
# conduct_one_big_run([0,1,2,3,42,99,100,132,420,999], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
