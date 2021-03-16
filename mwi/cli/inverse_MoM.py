from mwi.util import read_config
from mwi.util import model
from mwi.util.rx import MeasurementSurface

def inverse_MoM(model_config_file, prior_config_file, meas_config_file, output_folder, image_config_file, born_method):

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

    



def main():
    description ='''Microwave Inverse Using Method of Moments

    Example usage:
    inverse /models/homogeneous_ellipse/bkgd.json /models/homogeneous_ellipse/ellipse.json /models/homogeneous_ellipse/meas.json

    Calls the inverse function which:
        - read in configuration files
        - sets up gprMax forward solver files
        - calls gprMax forward solver
        - applies inverse algorithms to generate reconstructed image
    
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
    inverse_MoM(**vars(args))

    

if __name__ == "__main__":
    main()