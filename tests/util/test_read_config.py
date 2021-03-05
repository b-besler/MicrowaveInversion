import unittest
from mwi.util import read_config
import os

class TestReadModelConfig(unittest.TestCase):
    filePath = "example/model_config.json"
    
    def test_example_file_exists(self):
        self.assertTrue(os.path.exists(self.filePath))

    def test_read_example_file(self):
        output_dict = {
            'dx': 0.01, 
            'dy': 0.01, 
            'x1': -0.3, 
            'x2': 0.3, 
            'y1': -0.3, 
            'y2': 0.3,
            'name': "example_model",
            'objects': [
                {
                    'type': 'background', 'er': 1, 'sig': 0
                }, 
                {
                    'type': 'circle', 'er': 1.2, 'sig': 0.0, 'x0': 0.001, 'y0': 0.001, 'r': 0.075
                }, 
                {
                    'type': 'ellipse', 'er': 1.4, 'sig': 0.0, 'x0': 0.001, 'y0': 0.001, 'r1': 0.075, 'r2': 0.0375
                }
            ]
        }
        self.assertTrue(output_dict == read_config.read_model_config(self.filePath))

class TestReadMeasConfig(unittest.TestCase):
    filePath = "example/measurement_config.json"
    
    def test_example_file_exists(self):
        self.assertTrue(os.path.exists(self.filePath))

    def test_read_example_file(self):
        output_dict = {
            'measurement_surface': {
                'x0': 0.0,
                'y0': 0.0,
                'nr': 4, 
                'nt': 4, 
                'r': 0.15
                }, 
            'signal': {
                't0': 5e-09, 
                'f0': 1000000000.0, 
                'type': 'gaussiandot', 
                'amp': 1
                }
        }
        
        self.assertTrue(output_dict == read_config.read_meas_config(self.filePath))
