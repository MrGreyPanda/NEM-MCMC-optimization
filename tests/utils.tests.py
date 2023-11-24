import unittest
import numpy as np
import sys

sys.path.append('..')  # Add parent directory to module search path
from utils import create_connection_mat, create_real_knockdown_mat
from nem_order_mcmc import NEMOrderMCMC
import nem

class TestUtils(unittest.TestCase):        
    def test_create_real_knockdown_mat1(self):
        s_mat = [[0,1,1,0,1,0],
                 [0,0,1,0,1,0],
                 [0,0,0,0,1,0],
                 [0,0,1,0,1,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,1,0]]
        e_arr = [0, 1, 2, 3, 4, 5, 0]
        expected_result = np.array(
                          [[1, 1, 1, 0, 1, 0, 1],
                           [0, 1, 1, 0, 1, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0]])
        result = create_real_knockdown_mat(s_mat, e_arr)
        self.assertTrue(np.array_equal(result, expected_result), f"Expected: \n{expected_result}\n got: \n{result}")
        
class TestMCMC(unittest.TestCase): 
    def test_mcmc_run(self):
        s_mat = [[0,1,1,0,1],
                 [0,0,1,0,1],
                 [0,0,0,0,1],
                 [0,0,1,0,1],
                 [0,0,0,0,0]]
        num_s = 5
        num_e = 8
        e_arr = [0, 1, 2, 3, 0, 1, 4, 2]
        errors = [0.05, 0.1]
        my_nem = nem.NEM(adj_matrix=s_mat, end_nodes=e_arr, errors=errors, num_s=num_s, num_e=num_e)
        order = np.array([4,3,2,1,0])
        mcmc_nem = NEMOrderMCMC(my_nem, perm_order=order)
        gamma = 2.0 * num_s / num_e
        score, best_dag = mcmc_nem.method(n_iterations=110, gamma=gamma, seed=42)
        best_order = mcmc_nem.best_order
        print(f"Best order: {best_order}\nReal order: {my_nem.real_parent_order}\nObserved order: {my_nem.obs_parent_order}")
        print(f"Infered Order Score: {score}")
        print(f"Real Order Score: {my_nem.real_order_ll}, Real Score: {my_nem.real_ll}")
        print(f"Observed Order Score: {my_nem.obs_order_ll}, Observed Score: {my_nem.obs_ll}")
        

if __name__ == '__main__':
    unittest.main()