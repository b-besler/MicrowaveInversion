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
import mwi.util.dispersive_models as er_model

def sim_measurement(config_folder, material_file, output_folder, verbose, number):
    for i in range(int(number)):
        print("Creating simulated data")
        # read in configuration files from folder
        object_config, background_config, meas_config, image_config = read_config.read_config_folder(config_folder)

        # read in materials and update object definition as needed
        materials = er_model.read_materials(material_file)
        er_model.assign_properties(object_config['objects'], materials, image_config['recon_freq'])
        read_config.write_model_config(object_config, os.path.join(config_folder, 'object_updated'))
        # save updated model


        # get signal amplitude (current)
        current = image_config["current"]["real"] + 1j * image_config["current"]["imag"]
        # get signal noise level
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

        np.save(os.path.join(output_folder, obj_model.name + '_actual_er.npy'), obj_model.get_image('er'))
        np.save(os.path.join(output_folder, obj_model.name + '_actual_sig.npy'), obj_model.get_image('sig'))
        
        # do forward solve
        (inc_field, total_field, rx_scatter) = mom.sim_homogeneous_background(backgnd_model, obj_model, current)
        
        # do some formatting of fields
        rx_scatter = np.ravel(rx_scatter)
        rx_scatter = mom.add_noise(rx_scatter, noise)

        json_data = {}
        json_data['real'] = rx_scatter.real.tolist()
        json_data['imag'] = rx_scatter.imag.tolist()

        with open(os.path.join(output_folder, obj_model.name + '_scattered' + '_' + str(i) +'.json'), 'w') as file:
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
    parser.add_argument('config_folder', help='Folder with configuration files (.json)')
    parser.add_argument('material_file', help ='Folder with material files (.json)')
    parser.add_argument('output_folder', help='Folder to place outputs')
    parser.add_argument('-v','--verbose', help="Verbose output?", required = False, action='store_true')
    parser.add_argument('-n', '--number', help="Number of measurements to generate", required = False, default = 1)

    # Parse args and display
    args = parser.parse_args()

    # Run program
    sim_measurement(**vars(args))

    

if __name__ == "__main__":
    main()