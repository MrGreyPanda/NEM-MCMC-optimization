from nem import NEM
import utils
import numpy as np
from methods import InverseMethod
import wandb
import os
import random as rnd

def gen_thesis_data():
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
    # e_vec = [2,3,0,4,0,1,5,3,2,4,1,2,3,4,5,0,0,1,3,4,5,1,2,4,5,0,1,2,3,4,5,1,2,0,0,0,1,2,3,4,5]
    num_e = len(e_vec)
    nem = NEM(adj_mat, e_vec, errors, num_s, num_e)
    init_guess = utils.initial_order_guess(nem.observed_knockdown_mat)
    inv_method = InverseMethod(init_guess, num_s, num_e, nem.U, nem.get_score_tables(nem.observed_knockdown_mat))
    weights, ll = inv_method.optimize()
    print(f"Comparison: real_ll: {nem.obs_ll} - inferred_ll: {ll}")
    print(f"Hamming Distance: {np.sum(np.abs(weights - adj_mat))}")
    print(f"weights:\n{weights}")
    # print(weights)

def conduct_var_e_genes_experiments():
    wandb.login()
    seed = 42
    rnd.seed(seed)
    # for i in range(19):
    print(f"Network 11:")
    curr_dir = os.getcwd()
    network_nr = 11
    network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
    
    num_e_max = num_s * 30
    end_nodes = []
    for _ in range(num_s - 1):
            end_nodes.append(rnd.randint(0, num_s-1))
    seed = 42
    run = wandb.init(
    # Set the project where this run will be logged
    project="MCMC-NEM",
    # Track hyperparameters and run metadata
    config={
        "seed": seed,
        "num_s": num_s,
        "num_e_max": num_e_max,
        "adj_matrix": adj_matrix,
        "errors": errors
    })
    
    print(f"num_s: {num_s}")
    for j in range(num_s, num_e_max, 1):
        num_e = j
        print(f"num_e: {num_e}")
        end_nodes.append(rnd.randint(0, num_s-1))
        
        curr_nem = NEM(adj_matrix, end_nodes, errors, num_s, num_e)
        permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
        inv_method = InverseMethod(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
        weights, ll = inv_method.optimize()
        wandb.log({"Comparison": curr_nem.obs_ll - ll})
        wandb.log({"Hamming Distance": np.sum(np.abs(weights - adj_matrix))})
        wandb.log({"num_e": num_e})
        print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll}")
        print(f"Hamming Distance: {np.sum(np.abs(weights - adj_matrix))}")
        # print(f"weights:\n{weights}")


def conduct_fixed_e_genes_experiments():
    # max_iter = 1000
    # rel_diff = 1e-3
    # seed = 42
    # run = wandb.init(
    # # Set the project where this run will be logged
    # project="MCMC-NEM",
    # # Track hyperparameters and run metadata
    # config={
    #     "max_iterations:": max_iter,
    #     "rel_diff": rel_diff,
    #     "seed": seed,
    # })
    rnd.seed(42)
    for i in range(19):
        print(f"Network {i}:")
        curr_dir = os.getcwd()
        network_nr = i
        network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
        adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
        end_nodes.append(rnd.randint(0, num_s-1))
        curr_nem = NEM(adj_matrix, end_nodes, errors, num_s, num_e)
        permutation_order = utils.initial_order_guess(curr_nem.observed_knockdown_mat)
        inv_method = InverseMethod(permutation_order, num_s, num_e, curr_nem.U, curr_nem.get_score_tables(curr_nem.observed_knockdown_mat))
        weights, ll = inv_method.optimize()
        print(f"Comparison: real_ll: {curr_nem.obs_ll} - inferred_ll: {ll}")
        print(f"Hamming Distance: {np.sum(np.abs(weights - adj_matrix))}")
        # print(f"weights:\n{weights}")

# conduct_var_e_genes_experiments()
conduct_fixed_e_genes_experiments()
