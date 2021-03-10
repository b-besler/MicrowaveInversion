import argparse
import numpy as np
import os

from mwi.util import read_config
from mwi.util import model
from mwi.util.rx import MeasurementSurface
import mwi.util.sim as sim
import mwi.util.read_gprMax as read_gprMax


def inverse(model_config_file, prior_config_file, meas_config_file, output_folder, image_config_file):
    # Read in .json configuration data
    meas_data = read_config.read_meas_config(meas_config_file)
    model_data = read_config.read_model_config(model_config_file)
    prior_data = read_config.read_model_config(prior_config_file)
    image_data = read_config.read_domain_config(image_config_file)

    rx = MeasurementSurface(meas_data["measurement_surface"])
    rx.plot()
    rx.plot_discrete(0.01, 0.01)

    image_domain = model.ImageDomain(image_data)

    obj_model = model.Model(model_data, rx, image_domain)

    obj_model.plot_er()
    
    src = sim.Source(meas_data["signal"])
    sim.make(obj_model, src, output_folder)
    num_run = sim.run(os.path.join(os.path.abspath(output_folder), obj_model.name))

    (data_t, t, data_f, freq) = read_gprMax.read_out_data(os.path.join(output_folder, obj_model.name, obj_model.name + "_output"))
    (field_t, field_time, field_f, field_freq) = read_gprMax.read_snapshots(os.path.join(output_folder, obj_model.name))

def main():
    description ='''Microwave Inverse

    Example usage:
    inverse /models/homogeneous_ellipse/bkgd.json /models/homogeneous_ellipse/ellipse.json /models/homogeneous_ellipse/meas.json

    Calls the inverse function which:
        - read in configuration files
        - sets up gprMax forward solver files
        - calls gprMax forward solver
        - applies inverse algorithms
    
    #TODO
    Inverse solve parameters can be varied to include: 
        --born_type (born, iter, distort) - Born approximation, Iterative Born Method, and Distorted Iterative Born Method
    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="inverse",
        description=description
    )
    parser.add_argument('model_config_file', help='.json file with model configuration')
    parser.add_argument('prior_config_file', help='.json file with a priori model configuration')
    parser.add_argument('meas_config_file', help='.json file with measurement setup configuration')
    parser.add_argument('image_config_file', help='.json file with image domain configuration')
    parser.add_argument('output_folder', help='folder to place outputs, including simulation files')

    # Parse args and display
    args = parser.parse_args()

    # Run program
    inverse(**vars(args))

    

if __name__ == "__main__":
    main()