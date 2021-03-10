import os
import numpy as np
import unittest

import mwi.util.read_gprMax as read_gprMax

class TestReadData(unittest.TestCase):
    output_folder = os.path.join("example", "gprMax", "output")
    snapshot_folder = os.path.join("example", "gprMax")

    def test_output_folder_exist(self):
        self.assertTrue(os.path.exists(self.output_folder))
    
    def test_snapshot_folder_exist(self):
        self.assertTrue(os.path.exists(self.output_folder))

    def test_read_out_data(self):
        (data, t, data_f, f) = read_gprMax.read_out_data(self.output_folder)

        self.assertTrue(data.shape == (1,4,213))
        self.assertTrue(data_f.shape == (1,4,106))
        self.assertTrue(t.shape == (213,))
        self.assertTrue(f.shape == (106,))
        #TODO more tests such a recursion (i.e. are the data matrices the same) or calculate time vector

    def test_read_snapshot_data(self):
        (data, t, data_f, f) = read_gprMax.read_snapshots(self.snapshot_folder)

        self.assertTrue(data.shape == (1,3,20,20))
        self.assertTrue(data_f.shape == (1,1,20,20))
        self.assertTrue(t.shape == (3,))
        self.assertTrue(f.shape == (1,))
    