import os
import numpy as np
import unittest

import mwi.util.read_gprMax as read_gprMax

class TestReadData(unittest.TestCase):
    output_folder = os.path.join("example", "gprMax", "output")

    def test_output_folder_exist(self):
        self.assertTrue(os.path.exists(self.output_folder))

    def test_read_out_data(self):
        (data, t, data_f, f) = read_gprMax.read_out_data(self.output_folder)

        self.assertTrue(data.shape == (1,4,213))
        self.assertTrue(data_f.shape == (1,4,106))
        self.assertTrue(t.shape == (213,))
        self.assertTrue(f.shape == (106,))
        #TODO more tests such a recursion (i.e. are the data matrices the same) or calculate time vector
    