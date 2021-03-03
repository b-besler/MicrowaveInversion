import unittest
from mwi.util.rx import MeasurementSurface
from mwi.util.read_config import read_meas_config
import os

class TestRx(unittest.TestCase):
    example_file = "example/measurement_config.json"

    def test_example_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))

    def test_rx_class_init(self):
        meas_surf = MeasurementSurface(read_meas_config(self.example_file))

        self.assertTrue(meas_surf.x0 == 0.0)
        self.assertTrue(meas_surf.y0 == 0.0)
        self.assertTrue(meas_surf.ntx == 24)
        self.assertTrue(meas_surf.nrx == 48)
        self.assertTrue(meas_surf.r == 0.15)
    
    

