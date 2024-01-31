from nem import NEM
from utils import read_csv_to_adj, ancestor, transitive_reduction

def gen_thesis_data():
    σ = [3, 4, 1, 5, 1, 2, 6, 4, 3, 5]
    adj_mat = [[0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 1, 1],
               [1, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0,],
               [0, 0, 0, 0, 0, 0]]
    
    num_s = 6
    num_e = 10
    α = 0.05
    β = 0.1
    
     
