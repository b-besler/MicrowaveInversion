import numpy as np
import unittest
import os

import mwi.util.calc as calc
import mwi.util.model as model
from mwi.util.rx import MeasurementSurface
import mwi.util.read_config as read_config

class TestCalcHankInt(unittest.TestCase):
    example_file = "example/measurement_config.json"
    model_file = "example/model_config.json"
    domain_file = "example/image_domain.json"

    def setUp(self):
        self.rx = MeasurementSurface(read_config.read_meas_config(self.example_file)["measurement_surface"])
        self.domain = model.ImageDomain(read_config.read_domain_config(self.domain_file))
        self.model = model.Model(read_config.read_model_config(self.model_file), self.rx, self.domain)

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))
        self.assertTrue(os.path.exists(self.model_file))
        self.assertTrue(os.path.exists(self.domain_file))
    
    def test_calc_hank_int(self):
        hank_int = calc.hankel_integral(self.model)

        self.assertTrue(hank_int.shape[0] == self.model.nrx)
        self.assertTrue(hank_int.shape[1] == self.model.nf)
        self.assertTrue(hank_int.shape[2] == self.model.image_domain.y_cell.size)
        self.assertTrue(hank_int.shape[3] == self.model.image_domain.x_cell.size)


