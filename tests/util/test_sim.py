import unittest
import mwi.util.sim as sim
import mwi.util.read_config as read_config
import mwi.util.model as model
from mwi.util.rx import MeasurementSurface
import os
import h5py
import numpy as np


class TestSource(unittest.TestCase):
    example_file = "example/measurement_config.json"

    def test_file_exists(self):
        # test that file exists
        self.assertTrue(os.path.exists(self.example_file))
    
    def test_source_init(self):
        src = sim.Source(read_config.read_meas_config(self.example_file)["signal"])
        self.assertAlmostEqual(src.t0, 5e-9)
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

        
        files = [f for f in os.listdir("example/"+obj_model.name) if os.path.isfile(os.path.join("example/"+obj_model.name, f))]

        # check there are ntx simulation files
        in_files = [f for f in files if f.endswith('.in')]
        self.assertTrue(len(in_files) == obj_model.ntx)

        # read in new and old (example) files
        with open('example/example_model_Tx0.in', 'r') as file:
            prev = file.read()

        with open('example/example_model/example_model_Tx0.in', 'r') as file:
            new = file.read()

        
        # compare old and new files        
        self.assertTrue(prev == new)

        # remove new files
        for f in files:
            os.remove(os.path.join('example',obj_model.name, f))
        os.rmdir('example/example_model')
    
    def test_write_hdf5(self):
        obj_model = model.Model(read_config.read_model_config(self.model_file), self.rx, self.domain)
        data = obj_model.er

        keys = ("data", "rigidE", "rigidH")
        attrs = ( 'gprMax','dx_dy_dz','title')

        sim.write_hdf5(obj_model, 'example/test.hdf5', data)

        file_new = h5py.File('example/test.hdf5', 'r')
        file_prev = h5py.File('example/geometry.hdf5','r')

        if not all(k in file_new.keys() for k in keys):
            raise ValueError('Missing data group')
        
        if not all(k in file_new.attrs for k in attrs):
            raise ValueError('Missing data group')
            
        self.assertTrue(np.allclose(file_new["data"], file_prev["data"]))
        self.assertTrue(np.allclose(file_new["rigidE"], file_prev["rigidE"]))
        self.assertTrue(np.allclose(file_new["rigidH"], file_prev["rigidH"]))

        
        file_new.close()
        file_prev.close()
        os.remove('example/test.hdf5')
    
    def test_make_geometry(self):
        obj_model = model.Model(read_config.read_model_config(self.model_file), self.rx, self.domain)
        sim.make_geometry(obj_model, "example")

        with open(os.path.join("example", obj_model.name, obj_model.name + "_geometry.txt"), 'r') as file:
            prev = file.read()

        with open(os.path.join('example','example_model_geometry.txt'), 'r') as file:
            new = file.read()

        self.assertTrue(prev == new)



        
        





        
