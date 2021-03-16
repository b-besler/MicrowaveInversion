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


def inverse(model_config_file, prior_config_file, meas_config_file, output_folder, image_config_file, born_method):

    threshold = 0.001

    if not(born_method == 'born') and not(born_method == 'iterative') and not(born_method == 'distorted'):
        os.sys.exit('ERROR: method {} is not supported'.format(born_method))

    if born_method == 'born':
        niteration = 1
        born_method = ''
    else:
        niteration = 10

    print('Solving Microwave Inverse using Born ' + born_method.capitalize() + " method. With max {} iterations.".format(niteration))

    # Read in .json configuration data
    meas_data = read_config.read_meas_config(meas_config_file)
    model_data = read_config.read_model_config(model_config_file)
    prior_data = read_config.read_model_config(prior_config_file)
    image_data = read_config.read_domain_config(image_config_file)

    # initialize measurement surface using config data
    rx = MeasurementSurface(meas_data["measurement_surface"])
    #initialize imaging domain/ reconstruction parameters
    image_domain = model.ImageDomain(image_data)
    
    # initialize models (for "measured scattered fields")
    obj_model = model.Model(model_data, rx, image_domain)
    obj_model.plot_er()
    obj_model.plot_sig()
    bkgrd_model = model.Model(prior_data, rx, image_domain)
    
    # initialize source, make and run simulation
    src = sim.Source(meas_data["signal"])

    sim.make(obj_model, src, output_folder)
    sim.run(os.path.join(os.path.abspath(output_folder), obj_model.name))

    sim.make(bkgrd_model, src, output_folder)
    sim.run(os.path.join(os.path.abspath(output_folder), bkgrd_model.name))

    # read in "measured" data
    (total_data, _, total_data_f, _) = read_gprMax.read_out_data(os.path.join(output_folder, obj_model.name, obj_model.name + "_output"))
    (_, _, inc_data_f, freq) = read_gprMax.read_out_data(os.path.join(output_folder, bkgrd_model.name, bkgrd_model.name + "_output"))

    # calculate scattered fields and select measurements to use
    s_data_f = total_data_f - inc_data_f

    (_, f_indx) = calc.find_nearest(freq, bkgrd_model.image_domain.freq)
    # plt.plot(np.squeeze(np.abs(s_data_f[0, :, f_indx])))
    # plt.plot(np.squeeze(np.roll(np.abs(s_data_f[4, :, f_indx]), 4)))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[0, :, f_indx] - np.roll(s_data_f[4, :, f_indx], 4))))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[1, :, f_indx])))
    # plt.plot(np.squeeze(np.roll(np.abs(s_data_f[5, :, f_indx]), 4)))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[1, :, f_indx] - np.roll(s_data_f[5, :, f_indx], 4))))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[2, :, f_indx])))
    # plt.plot(np.squeeze(np.roll(np.abs(s_data_f[6, :, f_indx]), 4)))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[2, :, f_indx] - np.roll(s_data_f[6, :, f_indx], 4))))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[3, :, f_indx])))
    # plt.plot(np.squeeze(np.roll(np.abs(s_data_f[7, :, f_indx]), 4)))
    # plt.show()
    # plt.plot(np.squeeze(np.abs(s_data_f[3, :, f_indx] - np.roll(s_data_f[7, :, f_indx], 4))))
    # plt.show()

    s_data_f = calc.select_data(bkgrd_model, s_data_f, freq)

    (_,_,field_f, field_freq) = read_gprMax.read_snapshots(os.path.join(output_folder, obj_model.name))

    (_,f_indx) = calc.find_nearest(field_freq, obj_model.freq)
    
    plt.imshow(np.abs(np.squeeze(field_f[0,f_indx,:,:])))
    plt.show()
    plt.imshow(np.abs(np.squeeze(field_f[4,f_indx,:,:])))
    plt.show()

    plt.imshow((np.abs(np.squeeze(field_f[0,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[4,f_indx,:,:]))))))
    plt.show()
    plt.imshow((np.abs(np.squeeze(field_f[1,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[5,f_indx,:,:]))))))
    plt.show()
    plt.imshow((np.abs(np.squeeze(field_f[2,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[6,f_indx,:,:]))))))
    plt.show()
    plt.imshow((np.abs(np.squeeze(field_f[3,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[7,f_indx,:,:]))))))
    plt.show()

    plt.plot(np.squeeze(total_data[0,4,:]))
    plt.show()

    
    # calculate greens function
    hank_int = calc.hankel_integral(bkgrd_model)

    # plt.imshow((np.abs(np.squeeze(hank_int[0,:,:,:]) - np.rot90(np.rot90(np.squeeze(hank_int[4,0,:,:]))))))
    # plt.show()
    # plt.imshow((np.abs(np.squeeze(hank_int[1,:,:,:]) - np.rot90(np.rot90(np.squeeze(hank_int[5,0,:,:]))))))
    # plt.show()
    # plt.imshow((np.abs(np.squeeze(hank_int[2,:,:,:]) - np.rot90(np.rot90(np.squeeze(hank_int[6,0,:,:]))))))
    # plt.show()
    # plt.imshow((np.abs(np.squeeze(hank_int[3,:,:,:]) - np.rot90(np.rot90(np.squeeze(hank_int[7,0,:,:]))))))
    # plt.show()
    iter_model = bkgrd_model
    i = 0
    while True:

        # do iteration model
        
        iter_model.name = 'iteration' + str(i)
        iter_model.plot_er()
        sim.make(iter_model, src, output_folder)
        sim.run(os.path.join(os.path.abspath(output_folder), iter_model.name))

        (fields,_,field_f, field_freq) = read_gprMax.read_snapshots(os.path.join(output_folder, iter_model.name))

        (_,f_indx) = calc.find_nearest(field_freq, iter_model.freq)

        plt.plot(np.squeeze(fields[0,:,10,0]))
        plt.plot(np.squeeze(fields[4,:,10,19]))
        plt.show()

        print(np.mean(np.abs(np.squeeze(field_f[0,f_indx,:,:]))))
        plt.imshow(np.abs(np.squeeze(field_f[0,f_indx,:,:])))
        plt.show()
        plt.imshow(np.abs(np.squeeze(field_f[4,f_indx,:,:])))
        plt.show()

        plt.imshow((np.abs(np.squeeze(field_f[0,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[4,f_indx,:,:]))))))
        plt.show()
        plt.imshow((np.abs(np.squeeze(field_f[1,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[5,f_indx,:,:]))))))
        plt.show()
        plt.imshow((np.abs(np.squeeze(field_f[2,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[6,f_indx,:,:]))))))
        plt.show()
        plt.imshow((np.abs(np.squeeze(field_f[3,f_indx,:,:]) - np.rot90(np.rot90(np.squeeze(field_f[7,f_indx,:,:]))))))
        plt.show()

        plt.imshow(np.abs(np.squeeze(field_f[0,f_indx,:,:])))
        plt.show()
        plt.imshow(np.abs(np.squeeze(field_f[4,f_indx,:,:])))
        plt.show()


        # form data operator matrix
        data_op = calc.form_data_operator(iter_model, hank_int, field_f, field_freq)
        
        # make L-curve and find optimum
        print("Calculating L-Curve")
        (res_norm, soln_norm, gamma) = calc.L_curve(s_data_f, data_op, 10)
        (kappa, opt_gamma, gamma_idx) = calc.L_curve_knee(np.log10(res_norm), np.log10(soln_norm), gamma)

        # plot L-curve
        plt.loglog(res_norm, soln_norm)
        plt.loglog(res_norm[gamma_idx], soln_norm[gamma_idx],'r.')
        plt.title("L-Curve")
        plt.xlabel("Residual Norm")
        plt.ylabel("Solution Norm")
        plt.grid()
        plt.show()

        plt.semilogx(gamma, kappa)
        plt.semilogx(opt_gamma, kappa[gamma_idx])
        plt.show()

        # do inversion
        contrast = calc.solve_regularized(s_data_f, data_op, opt_gamma, np.identity((data_op.shape[1])))

        # calculated scattered fields
        s_data_calc = data_op @ contrast

        # calculate residuals and metric
        (_, rmse) = calc.residuals_percent(s_data_f, s_data_calc)
        print(f"Root mean square error: {rmse:0.2f}%")

        # form er image
        er = np.reshape(contrast.real +1, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
        indx = (er < 1)
        er[indx] = 1.0
        er_imag = np.reshape(contrast.imag, (iter_model.image_domain.y_cell.size, iter_model.image_domain.x_cell.size))
        indx = (er_imag < 0)
        er_imag[indx] = 0


        plt.imshow(er - np.rot90(np.rot90(er)))
        plt.show()

        plt.imshow(er - np.rot90(er))
        plt.show()

        plt.imshow(er_imag - np.rot90(np.rot90(er_imag)))
        plt.show()

        plt.imshow(er_imag - np.rot90(er_imag))
        plt.show()

        # compare er image to model
        (rsse_x, rsse_y) = obj_model.compare_to_image(er, 'er', True)
        print(f"Root of sum square error in x: {rsse_x:0.2f}")
        print(f"Root of sum square error in y: {rsse_y:0.2f}")

        # plot scattered fields
        plt.plot(np.abs(s_data_calc), label = "Calculated")
        plt.plot(np.abs(s_data_f), label = "Measured")
        plt.title("Scattered field magnitude")
        plt.xlabel("Rx/Tx Combination")
        plt.ylabel("Magnitude [A.U.]")
        plt.legend()
        plt.show()

        plt.plot(np.angle(s_data_calc), label = "Calculated")
        plt.plot(np.angle(s_data_f), label = "Measured")
        plt.title("Scattered field phase")
        plt.xlabel("Rx/Tx Combination")
        plt.ylabel("Phase [rad]")
        plt.legend()
        plt.show()

        # plot er image
        plt.imshow(er)
        plt.colorbar()
        plt.title("Reconstructed Permittivity")
        plt.show()

        plt.imshow(er_imag)
        plt.colorbar()
        plt.title("Reconstructed Imaginary Permittivity")
        plt.show()

        # plot model image
        obj_image = obj_model.get_image('er')
        plt.imshow(obj_image)
        plt.title("Input Permittivity")
        plt.show()

        # check whether to update model for next iteration
        if rmse < threshold:
            print(f'Residual RMSE threshold {threshold:0.2f} has been met. Terminating inversion with RMSE = {rmse:0.2f}')
            break
        
        i +=1 

        if i >= niteration:
            print('Max number of iterations met. Terminating inversion without reaching RMSE threshold.')
            break
        
        # update model
        iter_model.add_image(er, 'er')
        iter_model.add_image(er_imag, 'er_imag')
        
        plt.imshow(iter_model.get_image('er') - obj_model.get_image('er'))
        plt.show()


    


    
	
    

def main():
    description ='''Microwave Inverse

    Example usage:
    inverse /models/homogeneous_ellipse/bkgd.json /models/homogeneous_ellipse/ellipse.json /models/homogeneous_ellipse/meas.json

    Calls the inverse function which:
        - read in configuration files
        - sets up gprMax forward solver files
        - calls gprMax forward solver
        - applies inverse algorithms to generate reconstructed image
    
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
    parser.add_argument('output_folder', help='Folder to place outputs, including simulation files')
    parser.add_argument('-b','--born_method', help='Which Born method to use: born, iterative, distorted', required = False, default='born')

    # Parse args and display
    args = parser.parse_args()

    # Run program
    inverse(**vars(args))

    

if __name__ == "__main__":
    main()