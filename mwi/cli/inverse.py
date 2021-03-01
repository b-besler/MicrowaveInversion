import argparse

def inverse(model_config, prior_config, meas_config):
    print('Configuration file 1: ' + model_config)
    print('Configuration file 2: ' + prior_config)
    print('Measurement configuration: ' + meas_config)

def main():
    description ='''Microwave Inverse

    Example usage:
    inverse /models/homogeneous_ellipse/bkgd.json /models/homogeneous_ellipse/ellipse.json /models/homogeneous_ellipse/meas.json


    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="inverse",
        description=description
    )
    parser.add_argument('model_config', help='.json file with model configuration')
    parser.add_argument('prior_config', help='.json file with a priori model configuration')
    parser.add_argument('meas_config', help='.json file with measurement setup configuration')

    # Parse args and display
    args = parser.parse_args()

    # Run program
    inverse(**vars(args))

    

if __name__ == "__main__":
    main()