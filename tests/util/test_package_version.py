import unittest
import mwi

class TestPackageVersion(unittest.TestCase):
    def test_hasattr_version_info(self):
        self.assertTrue(hasattr(mwi, "version_info"))

    def test_hasattr__version__(self):
        self.assertTrue(hasattr(mwi, "__version__"))
        

if __name__ == '__main__':
    unittest.main()