import unittest
import mwi.util.model as model
from mwi.util.read_config import read_model_config
import numpy as np
import os

class TestModel(unittest.TestCase):
    example_file = "example/model_config_small.json"
    config = read_model_config(example_file)

    def test_example_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))

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

        



    

