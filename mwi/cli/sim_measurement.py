import argparse
import numpy as np
import os
import sys
import json

from mwi.util import read_config
from mwi.util import model
from mwi.util.rx import MeasurementSurface
import mwi.util.mom as mom
import mwi.util.calc as calc
import mwi.util.constants as constants

def sim_measurement(object_config_file, background_config_file, meas_config_file, image_config_file, output_folder, verbose):
    print("Creating simulated data")
    
    # load in .json configuration files into dicts
    meas_config = read_config.read_meas_config(meas_config_file)
    object_config = read_config.read_model_config(object_config_file)
    background_config = read_config.read_model_config(background_config_file)
    image_config = read_config.read_domain_config(image_config_file)

    current = image_config["current"]["real"] + 1j * image_config["current"]["imag"]
    noise = image_config["noise_level"]

    # initialize measurement surface using config data
    rx = MeasurementSurface(meas_config["measurement_surface"])

    #initialize imaging domain/ reconstruction parameters and alternative grid
    image_domain = model.ImageDomain(image_config, 0)
    alt_grid = model.ImageDomain(image_config, 0.02)
    
    # initialize models (for "measured scattered fields")
    obj_model = model.Model(object_config, rx, image_domain, alt_grid)
    backgnd_model = model.Model(background_config, rx, image_domain, alt_grid)

    if verbose:
        obj_model.plot_er()
        obj_model.plot_sig()
        backgnd_model.plot_er()
        backgnd_model.plot_er()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    np.save(os.path.join(output_folder, obj_model.name + '_actual_er1.npy'), obj_model.get_image('er'))
    np.save(os.path.join(output_folder, obj_model.name + '_actual_sig1.npy'), obj_model.get_image('sig'))
    
    # do forward solve
    (inc_field, total_field, rx_scatter) = mom.sim_homogeneous_background(backgnd_model, obj_model, current)
    
    # do some formatting of fields
    rx_scatter = np.ravel(rx_scatter)
    rx_scatter = mom.add_noise(rx_scatter, noise)

    json_data = {}
    json_data['real'] = rx_scatter.real.tolist()
    json_data['imag'] = rx_scatter.imag.tolist()

    with open(os.path.join(output_folder, obj_model.name + '_scattered.json'), 'w') as file:
        json.dump(json_data, file, indent=4)

def main():
    description ='''Create simulated microwave scattering data on measurement surface due to object model.

    Example usage:
    create_simulated_data object_model.json background_model.json measurement_data.json image.json output_folder

    Create simulated data at measurement locations for:
        - scattered fields (due to object)

    Saves in .json format for use in other problems (i.e. inverse solver
    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="sim_measurement",
        description=description
    )
    parser.add_argument('object_config_file', help='.json file with model configuration')
    parser.add_argument('background_config_file', help='.json file with a background configuration')
    parser.add_argument('meas_config_file', help='.json file with measurement setup configuration')
    parser.add_argument('image_config_file', help='.json file with image domain configuration')
    parser.add_argument('output_folder', help='Folder to place outputs')
    parser.add_argument('-v','--verbose', help="Verbose output?", required = False, action='store_true')

    # Parse args and display
    args = parser.parse_args()

    # Run program
    sim_measurement(**vars(args))

    

if __name__ == "__main__":
    main()