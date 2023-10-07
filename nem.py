import numpy as np
import utils
import os

class NEM:
    def __init__(self):
        # Get the current directory
        current_dir = os.getcwd()

        # Read the network.csv file
        with open(os.path.join(current_dir, 'network.csv'), 'r') as f:
            # Read the first line to get num_s and num_e
            self.num_s, self.num_e = map(int, f.readline().strip().split(','))

            # Create an empty adjacency matrix
            adj_matrix = np.zeros((self.num_s, self.num_s), dtype=int)

            # Read the connected s-points and update the adjacency matrix
            for line in f:
                s_points = list(map(int, line.strip().split(',')))
                if len(s_points) != 2:
                    break
                print(s_points)
                adj_matrix[s_points[0]-1, s_points[1]-1] = 1
                
            # Read the last line to get the end nodes
            end_nodes = np.array(list(map(int, line.split(','))))

        # Save the adjacency matrix and end nodes as class members
        self.s_mat = adj_matrix
        self.e_arr = end_nodes
        self.conn_mat = utils.create_connection_mat(self.s_mat, self.e_arr)
