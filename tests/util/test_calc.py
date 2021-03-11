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
        # TODO add recursive test to test funcitonality

class TestFindNearest(unittest.TestCase):
    def test_nearest_value(self):
        array = np.linspace(0, 10, num = 11)
        values = np.array([0.1, 5.6])

        (nearest_val, idx) = calc.find_nearest(array, values)

        print(nearest_val)
        print(idx)

        self.assertAlmostEqual(nearest_val[0], 0.0)
        self.assertAlmostEqual(nearest_val[1], 6.0)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], 6)

class TestFormData(unittest.TestCase):
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

    def test_form_data_operator(self):
        # test with zeros
        freq = np.linspace(1e8, 1e10, num = 100)
        e_field = np.zeros((4, 100, 20, 20))
        hank_int = np.zeros((4, 100, 20 ,20))
        data_operator = calc.form_data_operator(self.model, hank_int, e_field, freq)

        self.assertTrue(data_operator.shape == (3* 4* 1, 20* 20))
    # TODO add recursive test to test funcitonality

    def test_select_data(self):
        data = np.zeros((4,4,10))
        freq = np.linspace(1e8, 1e10, num = 10)
        data_out = calc.select_data(self.model, data, freq)
        self.assertTrue(data_out.shape == (3*4*1,))
    

class TestLCurve(unittest.TestCase):

    def test_L_curve(self):
        data_operator = np.array([
            [1+1j, 2+2j, 3+3j],
            [2+2j, 3+3j, 4+4j]
        ])

        x = np.array([1+1j, 2+2j])

        n = 10

        (res_norm, soln_norm, gamma) = calc.L_curve(x, data_operator, n)

        self.assertTrue(res_norm.shape == (n,))
        self.assertTrue(soln_norm.shape == (n,))
        self.assertTrue(gamma.shape == (n,))
        # TODO recursion test


    def test_form_hermitian(self):
        A = np.array([
            [1+1j, 2+2j, 3+3j],
            [2+2j, 3+3j, 4+4j]
        ])

        A_h = calc.form_hermitian(A)

        # hermitian is square, element xy is conjugate of element yx, diagonal is real
        self.assertTrue(A_h.shape == (3,3))
        self.assertTrue(A_h[0,0].imag == 0)
        self.assertTrue(A_h[1,1].imag == 0)
        self.assertTrue(A_h[2,2].imag == 0)
        self.assertAlmostEqual(A_h[0,1], np.conjugate(A_h[1,0]))
        self.assertAlmostEqual(A_h[2,1], np.conjugate(A_h[1,2]))
        self.assertAlmostEqual(A_h[0,2], np.conjugate(A_h[2,0]))

    def test_solve_regularized(self):
        A = np.array([
            [1+1j, 2+2j, 3+3j],
            [2+2j, 3+3j, 4+4j]
        ])

        x = np.array([1+1j, 2+2j])

        R = np.identity(3)
        gamma = 10

        y = calc.solve_regularized(x, A, gamma, R)

        self.assertTrue(y.shape == (3,))
        # TODO do simple expected result