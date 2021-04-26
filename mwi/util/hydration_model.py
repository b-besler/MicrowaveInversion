import argparse
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import os

import mwi.util.dispersive_models as er_model

def calc_phi_b(sex, weight, age):
    if sex == 'M':
        phi = 79.45 - 0.24 * weight - 0.15 * age
    elif sex == 'F':
        phi = 69.81 - 0.26 * weight - 0.12 * age
    else:
        sys.exit("Invalid sex: terminating program")
    return phi

def calc_delta_w_eff(sex, weight, age):
    phi_b_young = calc_phi_b(sex, weight, 34)
    phi_b_actual = calc_phi_b(sex, weight, age)
    return weight * (1 - phi_b_actual/phi_b_young)

def hydration_properties(folder, file_name, debye_data, slope_data, subject_info, freq):
    """ Adjusts baseline properties for age and applies dehydration due to weight loss """

    # create output folder if it doesn't exist
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(debye_data, 'r') as file:
        tissues = json.load(file)
    
    with open(slope_data,'r') as file:
        slope = json.load(file)
    
    with open(subject_info, 'r') as file:
        subject = json.load(file)
    
    phi_b = calc_phi_b(subject['sex'], subject['weight'], subject['age'])
    m = np.array([slope['er_s']['m'],slope['er_f']['m'],slope['sig_s']['m']])
    delta_w = np.array(subject['delta_w'])/subject['weight']*100
    data = {}
    i = 1
    for key in tissues:
        # load water content of tissue
        phi_i = tissues[key]['water']*100
        # load baseline properties
        D_i = np.array([tissues[key]['er_s'], tissues[key]['er_inf'], tissues[key]['sig_s']])
        # apply change in properties due to weight

        D = D_i + np.outer(phi_i/phi_b * delta_w, m)
        # store new properties in dict to be written
        data[key]={
            'er_s': D[0,0], 
            'er_inf': D[0,1], 
            'sig_s':D[0,2],
            'tau':tissues[key]['tau'],
            'index':er_model.material_indices[key]}
        i+=1
    
        data['FreeSpace'] = {
        'er_s': 1,
        'er_inf':0,
        'sig_s':0,
        'tau':0,
        'index':er_model.material_indices['FreeSpace']
    }

    # write out data
    with open(os.path.join(folder, file_name) +'.json', 'w') as file:
        json.dump(data, file, indent = 4)
    

def main():
    description ='''Generate dielectric properties 

    Example usage:
    property_hydration results male 65 50

    Calls the inverse function which:
        - reads in tissue data from database
        - fits debye model
        - saves debye parameters
    
    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="apply_hydration",
        description=description
    )
    parser.add_argument('folder', help='Folder to save data to')
    parser.add_argument("file_name", help="Name for file (no extension)")
    parser.add_argument('debye_data', help = 'Full path to debye model .json')
    parser.add_argument('slope_data', help='Full path to slope data .json')
    parser.add_argument('subject_info', help = '.json file with info on subject')
    parser.add_argument("freq", help = "Frequency point to evaluate at")

    # Parse args and display
    args = parser.parse_args()

    # Run program
    hydration_properties(**vars(args))


if __name__ == "__main__":
    main()