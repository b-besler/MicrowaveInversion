import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

from mwi.util import read_config
from mwi.util import model
from mwi.util.rx import MeasurementSurface
import mwi.util.sim as sim
import mwi.util.read_gprMax as read_gprMax
import mwi.util.calc as calc


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
    sim.run(os.path.join(os.path.abspath(output_folder), obj_model.name))

    bkgrd_model = model.Model(prior_data, rx, image_domain)
    bkgrd_model.plot_er()
    sim.make(bkgrd_model, src, output_folder)
    sim.run(os.path.join(os.path.abspath(output_folder), bkgrd_model.name))

    (_, _, total_data_f, _) = read_gprMax.read_out_data(os.path.join(output_folder, obj_model.name, obj_model.name + "_output"))
    (inc_data, _, inc_data_f, freq) = read_gprMax.read_out_data(os.path.join(output_folder, bkgrd_model.name, bkgrd_model.name + "_output"))
    (_, _, field_f, field_freq) = read_gprMax.read_snapshots(os.path.join(output_folder, bkgrd_model.name))

    s_data_f = total_data_f - inc_data_f
    s_data_f = calc.select_data(bkgrd_model, s_data_f, freq)
	
    hank_int = calc.hankel_integral(bkgrd_model)
    data_op = calc.form_data_operator(bkgrd_model, hank_int, field_f, field_freq)
	
    (res_norm, soln_norm, gamma) = calc.L_curve(s_data_f, data_op, 100)
    (opt_gamma, gamma_idx) = calc.L_curve_knee(np.log10(res_norm), np.log10(soln_norm), gamma)
	
    plt.loglog(res_norm, soln_norm)
    plt.loglog(res_norm[gamma_idx], soln_norm[gamma_idx], 'r.', label = "gamma = " + str(opt_gamma))
    plt.grid()
    plt.show()

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