import unittest
from mwi.util.image_domain import ImageDomain
import mwi.util.read_config as read_config
import os

class TestImageDomain(unittest.TestCase):
    example_file = "example/image_domain.json"

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))

    def test_domain_init(self):
        domain = ImageDomain(read_config.read_domain_config(self.example_file))
        
        self.assertAlmostEqual(domain.x1, -0.1)
        self.assertAlmostEqual(domain.x2, 0.1)
        self.assertAlmostEqual(domain.y1, -0.1)
        self.assertAlmostEqual(domain.y2, 0.1)
        self.assertAlmostEqual(domain.dx, 0.01)
        self.assertAlmostEqual(domain.dy, 0.01)
