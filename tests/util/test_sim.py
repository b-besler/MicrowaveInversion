import unittest
import mwi.util.sim as sim
import mwi.util.read_config as read_config
import mwi.util.model as model
from mwi.util.rx import MeasurementSurface
import os


class TestSource(unittest.TestCase):
    example_file = "example/measurement_config.json"

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))
    
    def test_source_init(self):
        src = sim.Source(read_config.read_meas_config(self.example_file)["signal"])
        self.assertAlmostEqual(src.t0, 1e-9)
        self.assertAlmostEqual(src.f0, 1e9)
        self.assertTrue(src.type == 'gaussiandot')
        self.assertAlmostEqual(src.amp, 1)

class TestSim(unittest.TestCase):
    example_file = "example/measurement_config.json"
    model_file = "example/model_config.json"
    domain_file = "example/image_domain.json"

    rx = MeasurementSurface(read_config.read_meas_config(example_file)["measurement_surface"])
    domain = model.ImageDomain(read_config.read_domain_config(domain_file))
    src = sim.Source(read_config.read_meas_config(example_file)["signal"])

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))
        self.assertTrue(os.path.exists(self.model_file))
        self.assertTrue(os.path.exists(self.domain_file))

    def test_make_sim(self):
        obj_model = model.Model(read_config.read_model_config(self.model_file), self.rx, self.domain)
        sim.make(obj_model, self.src, 'example' )

        with open('example/example_model_Tx0.in', 'r') as file:
            prev = file.read()

        with open('example/example_model/example_model_Tx0.in', 'r') as file:
            new = file.read()
        
        self.assertTrue(prev == new)



        
