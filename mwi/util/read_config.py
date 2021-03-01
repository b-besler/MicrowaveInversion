import json
import os

def read_model_config(filepath):
    """Read model configuration .json file. 
    
    Example usage: mwi.util.read_config("example/model_config.json")

    Makes sure that dx, dy, x1/2, y1/2, and background objects are there and valid.
    Returns full dictionary

    Args:
        - filepath (str): filepath to .json file from main directory

    Outputs:
        - data (dict): dictionary holding .json data
    """

    # check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError ('Configuration file ' + filepath + ' does not exist.')
    
    # load data
    with open(filepath) as file:
        data = json.load(file)

    # mandatory keys
    keys = ("dx", "dy", "x1", "x2", "y1", "y2", "objects")

    # check they are all there
    if not all(k in data for k in keys):
        raise ValueError ('Missing configuration variable, check file')
    
    # check objects have required data
    for obj in data["objects"]:
        keys = ("er", "sig", "type")
        if not any(k in obj for k in keys):
            raise ValueError('Missing object information in configuration file')

        # check variables
        if obj["er"] <= 0:
            raise ValueError('Permittivity must be greater than zero')

        if obj["sig"] < 0:
            raise ValueError('Conductivity must be zero or greater')

        if obj["type"] == "circle":
            # check circle has neccessary parameters
            keys = ("x0","y0", "r")
            if not any(k in obj for k in keys):
                raise ValueError('Circle object must have x0, y0, and r')
            # Check that circle is inside model domain
            if obj["x0"] - obj["r"] < data["x1"] or obj["x0"] + obj["r"] > data["x2"]:
                raise ValueError('Circle object must be within model domain - check x')
            if obj["y0"] - obj["r"] < data["y1"] or obj["y0"] + obj["r"] > data["y2"]:
                raise ValueError('Circle object must be within model domain - check y')

        elif obj["type"] == "ellipse":
            # check that ellipse has neccessary parameters
            keys = ("x0","y0", "r1", "r2")
            if not any(k in obj for k in keys):
                raise ValueError('Ellipse object must have x0, y0, r1, and r2')
            # check that ellipse is inside model domain
            if obj["x0"] - obj["r1"] < data["x1"] or obj["x0"] + obj["r1"] > data["x2"]:
                raise ValueError('Ellipse object must be within model domain - check x')
            if obj["y0"] - obj["r2"] < data["y1"] or obj["y0"] + obj["r2"] > data["y2"]:
                raise ValueError('Ellipse object must be within model domain - check y')

    # check variables
    if data["dx"] <= 0 or data["dy"] <= 0:
        raise ValueError('Discretization must be greater than 0')

    if data['x1'] >= data['x2'] or data['y1'] >= data['y2']:
        raise ValueError('Second (x,y) coordinate must be greater than first')

    return data

def read_meas_config(filepath):
    """Read measurement configuration .json file. 
    
    Example usage: mwi.util.read_config("example/measurement_config.json")

    Makes sure that dx, dy, x1/2, y1/2, and background objects are there and valid.
    Returns full dictionary

    Args:
        - filepath (str): filepath to .json file from main directory

    Outputs:
        - data (dict): dictionary holding .json data
    """

    # check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError('Configuration file ' + filepath + ' does not exist.')
    
    # load data
    with open(filepath) as file:
        data = json.load(file)

    # mandatory keys
    keys = ("measurement_surface", "signal")

    # check they are all there
    if not all(k in data for k in keys):
        raise ValueError('Missing configuration variable, check file')
    
    keys = ("nr","nt","r")
    # check measurement surface variables are there
    if not any(k in data["measurement_surface"] for k in keys):
        raise ValueError('Missing measurement surface variable in configuration file')

    keys = ("t0","f0","type","amp")
    # check signal variables are there
    if not any(k in data["signal"] for k in keys):
        raise ValueError ('Missing signal variable in configuration file')

    #check variables
    if not (int(data["measurement_surface"]["nr"]) == data["measurement_surface"]["nr"]) or not (int(data["measurement_surface"]["nt"]) == data["measurement_surface"]["nt"]):
        raise TypeError('Number of receivers or transmitter must be integers')

    if data["measurement_surface"]["nr"] <= 0 or data["measurement_surface"]["nt"] <= 0:
        raise ValueError('Number of receivers or transmitters must be greater than zero')

    if data["measurement_surface"]["r"] <=0:
        raise ValueError('Radius of measurement surface must be greater than zero')

    if data["signal"]["t0"] <=0:
        raise ValueError('Signal window length must be greater than 0')

    if data["signal"]["f0"] <=0:
        raise ValueError('Signal center frequency must be greater than 0')

    if data["signal"]["amp"] <=0:
        raise ValueError('Signal amplitude must be greater than 0')

    if not data["signal"]["type"] == 'gaussiandot':
        raise ValueError('Unsupported signal type: ' + data["signal"]["type"])

    return data



