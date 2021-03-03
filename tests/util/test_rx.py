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

    def test_calc_pos(self):
        x_pos = np.array([0.15, 0, -0.15, 0])
        y_pos = np.array([0, 0.15, 0, -0.15])

        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])
        self.assertTrue(np.allclose(np.array([x_pos, y_pos]), meas_surf.calc_pos(meas_surf.ntx)))
    
    def test_calc_discrete_pos(self):
        dx = 0.5
        dy = 0.5

        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])

        x = np.array([round((meas_surf.x0 + meas_surf.r)/dx)*dx, round(meas_surf.x0/dx)*dx, round((meas_surf.x0 - meas_surf.r)/dx)*dx, round(meas_surf.x0/dx)*dx])
        y = np.array([round((meas_surf.y0)/dy)*dy, round((meas_surf.y0 + meas_surf.r)/dy)*dy, round((meas_surf.y0)/dy)*dy, round((meas_surf.y0 - meas_surf.r)/dy)*dy])

        self.assertTrue(np.allclose(meas_surf.calc_pos_discrete(meas_surf.nrx, dx, dy), np.array([x,y])))

    def test_translate(self):
        x = 1
        y = 2
        
        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])
        meas_surf.translate(1, 2)
        self.assertTrue(x == meas_surf.x0)
        self.assertTrue(y == meas_surf.y0)
    
    def test_is_too_close(self):
        bool_matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        
        meas_surf = MeasurementSurface(read_meas_config(self.example_file)["measurement_surface"])

        self.assertTrue(np.all(meas_surf.is_too_close(pi/2) == bool_matrix))
    

    



    


