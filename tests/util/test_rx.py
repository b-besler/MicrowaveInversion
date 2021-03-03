import unittest
from mwi.util.rx import MeasurementSurface
from mwi.util.read_config import read_meas_config
import os
from math import pi
import numpy as np

class TestRx(unittest.TestCase):
    example_file = "example/measurement_config.json"

    def test_example_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))

    def test_rx_class_init(self):
        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])

        self.assertTrue(meas_surf.x0 == 0.0)
        self.assertTrue(meas_surf.y0 == 0.0)
        self.assertTrue(meas_surf.ntx == 4)
        self.assertTrue(meas_surf.nrx == 4)
        self.assertTrue(meas_surf.r == 0.15)
    
    def test_calc_angle(self):
        angle = np.array([0, pi/2, pi, 3*pi/2])
        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])
        self.assertTrue(np.allclose(angle, meas_surf.calc_angle(meas_surf.nrx)))

    



    


