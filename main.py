import nem
from nem_order_mcmc import NEMOrderMCMC, replica_exchange_method
import utils
import numpy as np
from DAGs import dot
from DAGs import graph
import os
import shutil
import random
import matplotlib.pyplot as plt

def initial_order_guess(observed_knockdown_mat):
    """
    Make an "educated" guess on the order of the nodes in the network.
    """
    num_s = observed_knockdown_mat.shape[0]
    sum_col = np.sum(observed_knockdown_mat, axis=1)
    order = np.arange(num_s)
    order = np.argsort(-sum_col)
    return order

def remove_old_output():
    files = ["output/output.pdf", "output/output.dot", "output/output.gv",
             "output/infer_closed.dot", "output/infer_closed.pdf", "output/infer_closed.gv",
             "output/infer_red.dot", "output/infer_red.pdf", "output/infer_red.gv",
             "output/real_red.dot", "output/real_red.pdf", "output/real_red.gv",
             "output/real_closed.pdf"]
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    if not os.path.exists("output"):
        os.mkdir("output")

def clean(curr_dir):
    
    directory = curr_dir+"/output"
    for filename in os.listdir(directory):
        if filename.endswith(".gv"):
            os.remove(os.path.join(directory, filename))

def output_handling(best_dag, adj_matrix, network_path, curr_dir):
    remove_old_output()
    best_nem = utils.ancestor(best_dag)
    output_path = os.getcwd()
    dot.generate_dot_from_matrix(best_nem, output_path + "/output/infer_closed.dot")
    graph.create_graph_from_dot(output_path + "/output/infer_closed.dot", output_path + "/output/infer_closed.pdf")
    real_red_mat = utils.transitive_reduction(adj_matrix)
    best_red = utils.transitive_reduction(best_dag)
    dot.generate_dot_from_matrix(best_red, output_path + "/output/infer_red.dot")
    graph.create_graph_from_dot(output_path + "/output/infer_red.dot", output_path + "/output/infer_red.pdf")
    shutil.copy(network_path + ".pdf",f"{curr_dir}/output/real_closed.pdf")
    shutil.copy(network_path + "_red.pdf",f"{curr_dir}/output/real_red.pdf")
    clean(curr_dir)
    

def main():
    curr_dir = os.getcwd()
    network_nr = 2
    network_path = f"{curr_dir}/DAGs/networks/network{network_nr}/network{network_nr}"
    adj_matrix, end_nodes, errors, num_s, num_e = utils.read_csv_to_adj(network_path + ".csv")
    my_nem = nem.NEM(adj_matrix, end_nodes, errors, num_s, num_e)
    permutation_order = initial_order_guess(my_nem.observed_knockdown_mat)
    #### MCMC METHOD ####
    gamma = 2.0 * float(my_nem.num_s) / float(my_nem.num_e)
    mcmc_nem = NEMOrderMCMC(my_nem, permutation_order)
    score, best_dag = mcmc_nem.method(n_iterations=520, gamma=gamma, seed=42)
    score_list = mcmc_nem.curr_score_list
    best_order = mcmc_nem.best_order
    # #### REPLICA EXCHANGE METHOD ####
    # score, best_mcmc_dag = replica_exchange_method(nem=my_nem, init_order_guess=permutation_order, n_iter=300, n_exchange=20)
    # best_dag = best_mcmc_dag.best_dag
    # best_order = best_mcmc_dag.best_order
    #####
    print(f"Best order: {best_order}\nReal order: {my_nem.real_parent_order}\nObserved order: {my_nem.obs_parent_order}")
    print(f"Infered Order Score: {score}")
    print(f"Real Order Score: {my_nem.real_order_ll}, Real Score: {my_nem.real_ll}")
    print(f"Observed Order Score: {my_nem.obs_order_ll}, Observed Score: {my_nem.obs_ll}")
    
    # # print(f"Real Order Score: {my_nem.real_order_ll}, Real Score: {my_nem.real_ll}")
    output_handling(best_dag, adj_matrix, network_path, curr_dir)
    
    # plot the score_list
    # plt.plot(score_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Score')
    # plt.title('Score vs. Iteration')
    # plt.show()
    
if __name__ == "__main__":
    main()