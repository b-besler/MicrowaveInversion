## Create image from microwave scattering measurements by solving inversion problem

import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
import time
from scipy import linalg
import json
import matplotlib

from mwi.util import read_config
from mwi.util import model
from mwi.util.rx import MeasurementSurface
import mwi.util.mom as mom
import mwi.util.calc as calc
import mwi.util.constants as constants

def inverse_image(obj_config_file, background_config_file, prior_config_file, meas_config_file,  image_config_file, measurement_folder, output_folder,born_method, verbose):
    num = 0
    for measurement_file in os.listdir(measurement_folder):
        if measurement_file.endswith('.json'):

            print("Starting inverse")
            rmse_old = 1e20
            threshold = 0.05
            niteration = 50
            
            if born_method == "born":
                niteration = 1
                print("Solving inverse with Method of Moments and 1st Order Born Approximation")
            elif born_method == "iterative":
                print(f"Solving inverse with Method of Moments and Born Iterative Method. {niteration:d} iterations.")
            elif born_method == "distorted":
                print(f"Solving inverse with Method of Moments and Distorted Born Iterative Method. {niteration:d} iterations.")
            else:
                print(f"Method {born_method} is unsupported. Terminating...")

            meas_config = read_config.read_meas_config(meas_config_file)
            prior_config = read_config.read_model_config(prior_config_file)
            image_config = read_config.read_domain_config(image_config_file)
            background_config = read_config.read_model_config(background_config_file)
            object_config = read_config.read_model_config(obj_config_file)

            current = image_config["current"]["real"] + 1j* image_config["current"]["imag"]

            # initialize measurement surface using config data
            rx = MeasurementSurface(meas_config["measurement_surface"])
            #initialize imaging domain/ reconstruction parameters and alternative grid
            image_domain = model.ImageDomain(image_config, 0)
            alt_grid = model.ImageDomain(image_config, 0.02)
            
            # initialize models (for "measured scattered fields")
            obj_model = model.Model(object_config, rx, image_domain, alt_grid)
            backgnd_model = model.Model(background_config, rx, image_domain, alt_grid)
            iter_model = model.Model(prior_config, rx, image_domain, alt_grid)
            final_model = model.Model(prior_config, rx, image_domain, alt_grid)

            if verbose:
                plt.imshow(iter_model.get_image('er'))
                plt.colorbar()
                plt.title("Permittivity")
                plt.show()
                plt.imshow(iter_model.get_image('sig'))
                plt.colorbar()
                plt.title("Conductivity")
                plt.show()
                plt.imshow(np.real(iter_model.get_image('contrast')))
                plt.title("Real part of contrast")
                plt.show()
                plt.imshow(np.imag(iter_model.get_image('contrast')))
                plt.title("Imaginary part of contrast")
                plt.show()

            # do forward solve to get incident fields
            (inc_field, _,_) = mom.sim_homogeneous_background(backgnd_model, iter_model, current)
            inc_field = np.expand_dims(inc_field, axis =1)

            # load measurement data
            with open(os.path.join(measurement_folder,measurement_file)) as json_file:
                data = json.load(json_file)
            
            rx_scatter = np.array(data['real']) + 1j*np.array(data['imag'])

            # do 1st order inversion
            # calculate homogeneous greens function
            hank_int = mom.calc_hankel_integral(backgnd_model)

            # calculate data operator
            data_op = mom.form_greens_operator(backgnd_model, hank_int, inc_field)

            # do L-Curve
            (res_norm, soln_norm, gamma) = calc.L_curve(rx_scatter, data_op, 100)
            (kappa, opt_gamma, gamma_idx) = calc.L_curve_knee(np.log10(res_norm), np.log10(soln_norm), gamma)
            opt_gamma = opt_gamma *1000


            # plot L-curve
            if verbose:
                plt.loglog(res_norm, soln_norm)
                plt.loglog(res_norm[gamma_idx], soln_norm[gamma_idx],'r.')
                plt.title("L-Curve")
                plt.xlabel("Residual Norm")
                plt.ylabel("Solution Norm")
                plt.grid()
                plt.show()

                plt.semilogx(gamma, kappa)
                plt.semilogx(opt_gamma, kappa[gamma_idx],'r.')
                plt.title("Curvature of L-Curve")
                plt.show()

            contrast = calc.solve_regularized(rx_scatter.flatten(), data_op, opt_gamma, np.identity((data_op.shape[1])))

            contrast = contrast*iter_model.er_b + iter_model.er_b
            er = np.reshape(contrast.real, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
            er_imag = np.reshape(-contrast.imag, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))

            (rsse_x, rsse_y, rsse_profile, rsse_total) = obj_model.compare_to_image(er.real, 'er', False)
            (_, _, rsse_profile_imag, rsse_total_imag) = obj_model.compare_to_image(er_imag, 'er_imag', False)
            print(f"Root of sum square error in x: {rsse_x:0.2f}")
            print(f"Root of sum square error in y: {rsse_y:0.2f}")
            print(f"Root of sum square error over image: {rsse_total:0.2f}")
            print(f"Root of sum square error over profile: {rsse_profile:0.2f}")

            with open(f'{os.path.join(output_folder,obj_model.name + "_" + born_method + "_" + str(num)) }.csv','w') as file:
                file.write("Iteration, RRE, MSE Profile - Real,  MSE Image - Real, MSE Profile - Imag, MSE Image - Imag\n")

            print("Updating model...")

            iter_model.write_image(er,'er')
            iter_model.write_image(er_imag,'er_imag')

            final_model.write_image(er,'er')
            final_model.write_image(er_imag,'er_imag')
            if verbose:
                plt.imshow(iter_model.get_image('er_imag'))
                plt.title("er''")
                plt.colorbar()
                plt.show()


            iteration = 1
            while True:
                # Threshold check is at the top of loop before next iteration
                # Later loop breaks if iterations reached or threshold is met

                (_, fields,_) = mom.sim_homogeneous_background(backgnd_model, iter_model, current)
                fields = np.expand_dims(fields, axis =1)

                rx_scatter_sim = mom.calc_rx_scatter(iter_model,fields)

                (_, rmse) = calc.residuals_percent(rx_scatter.flatten(), rx_scatter_sim.flatten())

                with open(f'{os.path.join(output_folder,obj_model.name + "_" + born_method) }.csv','a') as file:
                    file.write(f"{iteration}, {rmse:0.4f}, {rsse_profile:0.4f}, {rsse_total:0.4f},{rsse_profile_imag:0.4f}, {rsse_total_imag:0.4f} \n")

                print(f"Root mean square error residuals: {rmse:0.4f}")
                
                # check whether to update model for next iteration
                if rmse < threshold:
                    print(f'Residual RMSE threshold {threshold:0.2f}% has been met. Terminating inversion with RMSE = {rmse:0.2f}%')
                    # update final model to current model
                    final_model.write_image(iter_model.get_image('er'),'er')
                    final_model.write_image(iter_model.get_image('er_imag'), 'er_imag')
                    break
                    
                if rmse > rmse_old:
                    print(f"RRE increased. Terminating with RRE = {rmse_old:0.4f}")
                    # final model is previous model (already assigned)
                    break

                if iteration >= niteration:
                    print('Max number of iterations met. Terminating inversion without reaching RMSE threshold.')
                    # update final model to current model
                    final_model.write_image(iter_model.get_image('er'),'er')
                    final_model.write_image(iter_model.get_image('er_imag'), 'er_imag')
                    break

                if born_method == 'iterative':
                    # Iterative method assumes homogeneous and uses measured scattered field
                    greens = hank_int
                    data_op = mom.form_greens_operator(backgnd_model, greens, fields)
                    meas_data = rx_scatter.flatten()
                elif born_method == 'distorted':
                    # Distorted method assumes inhomogeneous and uses the scattered field error
                    greens = mom.greens_from_fields(iter_model, fields, current)
                    data_op = mom.form_greens_operator(iter_model, greens, fields)
                    meas_data = rx_scatter.flatten() - rx_scatter_sim.flatten()


                (res_norm, soln_norm, gamma) = calc.L_curve(meas_data, data_op, 100)
                (kappa, opt_gamma, gamma_idx) = calc.L_curve_knee(np.log10(res_norm), np.log10(soln_norm), gamma)
                #opt_gamma = opt_gamma *100
                #opt_gamma = 10**opt_gamma*1.10
                opt_gamma = opt_gamma * (1 + pow(10, 3 - 0.5*iteration))

                contrast = calc.solve_regularized(meas_data, data_op, opt_gamma, np.identity((data_op.shape[1])))
                contrast = np.reshape(contrast, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))

                # form er image
                if born_method == "iterative":
                    contrast = contrast * iter_model.er_b + iter_model.er_b
                    er = np.reshape(contrast.real, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
                    er_imag = -np.reshape(contrast.imag, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
                else:
                    contrast = iter_model.get_image('contrast') + np.reshape(contrast, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
                    contrast = contrast * iter_model.er_b + iter_model.er_b
                    er_imag = -contrast.imag
                    er = contrast.real

                # compare er image to model
                (rsse_x, rsse_y, rsse_profile, rsse_total) = obj_model.compare_to_image(er.real, 'er', False)
                (rsse_x_imag, rsse_y_imag, rsse_profile_imag, rsse_total_imag) = obj_model.compare_to_image(er_imag, 'er_imag', False)
                if verbose:
                    print(f"Root of sum square error in x: {rsse_x:0.2f}")
                    print(f"Root of sum square error in y: {rsse_y:0.2f}")
                    print(f"Root of sum square error over image: {rsse_total:0.2f}")

                iteration += 1

                print("Updating model...")

                # keep track of previous model (have to do another simluation to see if residuals increase)
                final_model.write_image(iter_model.get_image('er'),'er')
                final_model.write_image(iter_model.get_image('er_imag'), 'er_imag')

                # update current iteration model
                iter_model.write_image(er,'er')
                iter_model.write_image(er_imag,'er_imag')

                # keep track of previous rre
                rmse_old = rmse
            
            # plot final image
            if verbose:
                print("Showing final reconstruction")
                plt.imshow(final_model.get_image('er'))
                plt.title("Final reconstructed permittivity")
                plt.colorbar()
                plt.show()

                plt.imshow(final_model.get_image('sig'))
                plt.title("Final reconstructed conductivity")
                plt.colorbar()
                plt.show()

                obj_model.compare_to_image(final_model.get_image('er'), 'er', True)
                obj_model.compare_to_image(final_model.get_image('sig'), 'sig', True)

            (_, fields,_) = mom.sim_homogeneous_background(backgnd_model, final_model, current)
            fields = np.expand_dims(fields, axis =1)

            rx_scatter_sim = mom.calc_rx_scatter(final_model,fields)

            if verbose:

                plt.plot(np.imag(rx_scatter_sim.flatten()), label = 'Simulated')
                plt.plot(np.imag(rx_scatter.flatten()), label = 'Measured')
                plt.title("Final Scattered Fields - Imaginary Component")
                plt.ylabel("Magnitude [ ]")
                plt.legend()
                plt.show()
                plt.plot(np.real(rx_scatter_sim.flatten()), label = 'Simulated')
                plt.plot(np.real(rx_scatter.flatten()), label = 'Measured')
                plt.title("Final Scattered Fields - Real Component")
                plt.ylabel("Magnitude [ ]")
                plt.legend()
                plt.show()

            np.save(os.path.join(output_folder, iter_model.name + '_' + born_method + '_er_' + str(num) + '.npy'), final_model.get_image('er'))
            np.save(os.path.join(output_folder, iter_model.name + '_' + born_method + '_sig_' + str(num) + '.npy'), final_model.get_image('sig'))
            #plt.imsave(os.path.join(output_folder, iter_model.name + '_' + born_method + '_er.png'), final_model.get_image('er'))
            #plt.imsave(os.path.join(output_folder, iter_model.name + '_' + born_method + '_sig.png'), final_model.get_image('sig'))
            
            rx_scatter_sim = np.ravel(rx_scatter_sim)
            json_data = {}
            json_data['real'] = rx_scatter_sim.real.tolist()
            json_data['imag'] = rx_scatter_sim.imag.tolist()

            with open(os.path.join(output_folder, "reconstructed_measurements_" + str(num) + ".json"), 'w') as file:
                json.dump(json_data, file, indent=4)
            
            num += 1 

    

def main():
    description ='''Create dielectric image from microwave scattering measurements

    Example usage:
    create_image

    Uses a priori estimate and microwave scattering measurements to create image  
    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="inverse",
        description=description
    )
    parser.add_argument('obj_config_file', help='.json file with object model configuration')
    parser.add_argument('background_config_file', help='.json file with background model configuration')
    parser.add_argument('prior_config_file', help='.json file with a priori model configuration')
    parser.add_argument('meas_config_file', help='.json file with measurement setup configuration')
    parser.add_argument('image_config_file', help='.json file with image domain configuration')
    parser.add_argument('measurement_folder', help = ".json file with scatter measurements")
    parser.add_argument('output_folder', help='Folder to place outputs, including simulation files')
    parser.add_argument('-b','--born_method', help='Which Born method to use: born, iterative, distorted', required = False, default='born')
    parser.add_argument('-v','--verbose', help="Verbose output?", required = False, action='store_true')

    # Parse args and display
    args = parser.parse_args()

    # Run program
    inverse_image(**vars(args))

    

if __name__ == "__main__":
    main()