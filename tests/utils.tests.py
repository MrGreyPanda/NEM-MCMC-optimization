import unittest
import numpy as np
import sys

sys.path.append('..')  # Add parent directory to module search path
from utils import create_connection_mat, create_real_knockdown_mat

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

if __name__ == '__main__':
    unittest.main()