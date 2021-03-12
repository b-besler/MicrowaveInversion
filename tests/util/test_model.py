import unittest
import numpy as np
import os
from matplotlib import pyplot as plt

import mwi.util.model as model
from mwi.util.read_config import read_model_config
from mwi.util.read_config import read_domain_config
from mwi.util.read_config import read_meas_config
from mwi.util.rx import MeasurementSurface


class TestModel(unittest.TestCase):
    example_file = "example/model_config_small.json"
    example_file2 = "example/model_config.json"
    rx_file = 'example/measurement_config.json'
    domain_file = 'example/image_domain.json'

    rx_config = read_meas_config(rx_file)
    config = read_model_config(example_file)
    config2 = read_model_config(example_file2)
    domain_config = read_domain_config(domain_file)

    def setUp(self):
        self.rx_surf = MeasurementSurface(self.rx_config["measurement_surface"])
        self.image_domain = model.ImageDomain(self.domain_config)

    def test_example_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))
        self.assertTrue(os.path.exists(self.rx_file))
        self.assertTrue(os.path.exists(self.domain_file))

    def test_add_circle(self):
        # output image is more like a chamfered square with these settings but verified with compass
        output_image = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1.2,1.2,1.2,1.2,1,1,1,1],
            [1,1,1,1.2,1.2,1.2,1.2,1.2,1.2,1,1,1],
            [1,1,1,1.2,1.2,1.2,1.2,1.2,1.2,1,1,1],
            [1,1,1,1.2,1.2,1.2,1.2,1.2,1.2,1,1,1],
            [1,1,1,1.2,1.2,1.2,1.2,1.2,1.2,1,1,1],
            [1,1,1,1,1.2,1.2,1.2,1.2,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
        ])
        test_model = model.Model(self.config,[],[])
        image = np.ones((test_model.x_cell.size, test_model.y_cell.size))
        test_model.add_circle(self.config["objects"][1], image, self.config["objects"][1]["er"])
        self.assertTrue(np.amin(image) == 1)
        self.assertTrue(np.amax(image) == 1.2)
        self.assertTrue(np.allclose(image, output_image))

    def test_model_class_init(self):
        output_image = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1.2,1.2,1.2,1.2,1,1,1,1],
            [1,1,1,1.2,1.4,1.4,1.4,1.4,1.2,1,1,1],
            [1,1,1,1.4,1.4,1.4,1.4,1.4,1.4,1,1,1],
            [1,1,1,1.4,1.4,1.4,1.4,1.4,1.4,1,1,1],
            [1,1,1,1.2,1.4,1.4,1.4,1.4,1.2,1,1,1],
            [1,1,1,1,1.2,1.2,1.2,1.2,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1],
        ])
        test_model = model.Model(self.config,[],[])
        # test initialized variables and some properties
        self.assertTrue(test_model.dx == self.config["dx"])
        self.assertTrue(test_model.dy == self.config["dy"])
        self.assertTrue(test_model.x1 == self.config["x1"])
        self.assertTrue(test_model.x2 == self.config["x2"])
        self.assertTrue(test_model.x1 == self.config["y1"])
        self.assertTrue(test_model.x2 == self.config["y2"])
        self.assertTrue(test_model.x.size == 13)
        self.assertTrue(test_model.y.size == 13)
        self.assertTrue(test_model.x_cell.size == 12)
        self.assertTrue(test_model.y_cell.size == 12)
        # test background value
        self.assertTrue(test_model.er[0][0] == 1.0)
        # test for circle value
        self.assertTrue(np.any(test_model.er == 1.2))
        # test for ellipse value
        self.assertTrue(np.amax(test_model.er) == 1.4)
        # test full matrix
        self.assertTrue(np.allclose(test_model.er, output_image))
        # test for sigma
        self.assertTrue(np.allclose(test_model.sig, np.zeros((test_model.x_cell.size, test_model.x_cell.size))))

    def test_model_translate(self):
        dx = 0.5
        dy = -0.1
        test_model = model.Model(self.config,self.rx_surf,self.image_domain)

        test_model.translate(dx,dy)
        # test initialized variables and some properties
        self.assertTrue(test_model.dx == self.config["dx"])
        self.assertTrue(test_model.dy == self.config["dy"])
        self.assertTrue(test_model.x1 == (self.config["x1"] + dx))
        self.assertTrue(test_model.x2 == (self.config["x2"] + dx))
        self.assertTrue(test_model.y1 == (self.config["y1"] + dy))
        self.assertTrue(test_model.y2 == (self.config["y2"] + dy))
        self.assertTrue(test_model.rx.x0 == dx)
        self.assertTrue(test_model.rx.y0 == dy)
        self.assertAlmostEqual(test_model.image_domain.x1, dx -0.1)
        self.assertAlmostEqual(test_model.image_domain.y1, dy -0.1)
        self.assertAlmostEqual(test_model.image_domain.x2, dx +0.1)
        self.assertAlmostEqual(test_model.image_domain.y2, dy +0.1)
    
    def test_get_cross_section(self):
        image = np.array([
            [1,2,3,4,5,6],
            [1,2,3,4,5,6],
            [2,3,3,4,5,6],
            [1,2,3,4,5,6],
            [1,2,3,4,5,6]
        ])

        x_cross = model.Model.get_cross_section(image, 0)

        self.assertTrue(np.allclose(x_cross, np.array([2,3,3,4,5,6])))

        y_cross = model.Model.get_cross_section(image,1)
        self.assertTrue(np.allclose(y_cross, np.ones(5) *4))
        # pass 1D array
        self.assertRaises(ValueError, model.Model.get_cross_section, np.ones(7), 1)
        # improper index
        self.assertRaises(ValueError, model.Model.get_cross_section, image, 2)

    def test_model_compare_cross_section(self):
        test_model = model.Model(self.config2,self.rx_surf,self.image_domain)

        image = np.ones((20, 20))

        (rsse_x, rsse_y) = test_model.compare_to_image(image, 'er', False)

        self.assertAlmostEqual(rsse_x, np.sqrt(16*0.4**2))
        self.assertAlmostEqual(rsse_y, np.sqrt(8*0.4**2 + 8*0.2**2))

    def test_get_cross_section(self):
        image = np.array([
            [1.4,1.4,1.4,1.4],
            [1.4,1.4,1.4,1.4],
            [1.4,1.4,1.4,1.4],
            [1.4,1.4,1.4,1.4]
        ])

        test_model = model.Model(self.config,self.rx_surf,self.image_domain)
        er_image = test_model.get_image('er')

        self.assertTrue(np.allclose(er_image, image))

        










class TestImageDomain(unittest.TestCase):
    example_file = "example/image_domain.json"

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))

    def test_domain_init(self):
        domain = model.ImageDomain(read_domain_config(self.example_file))
        
        self.assertAlmostEqual(domain.x1, -0.1)
        self.assertAlmostEqual(domain.x2, 0.1)
        self.assertAlmostEqual(domain.y1, -0.1)
        self.assertAlmostEqual(domain.y2, 0.1)
        self.assertAlmostEqual(domain.dx, 0.01)
        self.assertAlmostEqual(domain.dy, 0.01)
        self.assertAlmostEqual(domain.x_size, 0.2)
        self.assertAlmostEqual(domain.y_size, 0.2)
        self.assertAlmostEqual(domain.freq.size, 1)
        self.assertAlmostEqual(domain.freq[0], 1e9)
        self.assertAlmostEqual(domain.ex_angle, 90 * np.pi /180)
        
    def test_domain_translate(self):
        domain = model.ImageDomain(read_domain_config(self.example_file))
        dx = 0.5
        dy = -0.4
        domain.translate(dx,dy)
        self.assertAlmostEqual(domain.x1, 0.4)
        self.assertAlmostEqual(domain.x2, 0.6)
        self.assertAlmostEqual(domain.y1, -0.5)
        self.assertAlmostEqual(domain.y2, -0.3)


    

