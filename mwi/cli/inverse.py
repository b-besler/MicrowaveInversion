import argparse
from mwi.util import read_config

def inverse(model_config, prior_config, meas_config):
    print(model_config)
    print(prior_config)
    print(meas_config)
    meas_data = read_config.read_meas_config(meas_config)
    model_data = read_config.read_model_config(model_config)
    print(meas_data)
    print(model_data)

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