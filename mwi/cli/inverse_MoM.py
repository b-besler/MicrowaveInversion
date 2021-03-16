

def inverse_MoM(model_config_file, prior_config_file, meas_config_file, output_folder, image_config_file, born_method):


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