import unittest
import mwi.util.sim as sim
import mwi.util.read_config as read_config
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

        
