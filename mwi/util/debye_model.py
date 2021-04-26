import argparse
import xlrd
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import minimize
import scipy.optimize as optimize
from matplotlib import pyplot as plt
import os
import json
import sys


import mwi.util.constants as constants
import mwi.util.dispersive_models as er_model

### Scripts for Debye models of tissue
def er_fit(x, m):
    return m * x + 1

def sig_fit(x, m):
    return m * x

def r_squared(popt, func, xdata, ydata):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    return 1 - (ss_res/ss_tot)

def fit_water(folder, file_name):
    file = os.path.join(folder,file_name) +'.json'
    print(file)
    with open(file) as json_file:
        tissues = json.load(json_file)

    n_tissues = len(tissues)
    water_content = np.zeros(n_tissues)
    er_s = np.zeros(n_tissues)
    er_f = np.zeros(n_tissues)
    sig_s = np.zeros(n_tissues)

    i = 0
    for key in tissues:
        water_content[i] = tissues[key]['water']
        er_s[i] = tissues[key]['er_s']
        er_f[i] = tissues[key]['er_inf']
        sig_s[i] = tissues[key]['sig_s']
        i+=1

    water_content *= 100

    er_s_opt, _ = optimize.curve_fit(er_fit, water_content, er_s)
    er_f_opt, _ = optimize.curve_fit(er_fit, water_content, er_f)
    sig_s_opt, _ = optimize.curve_fit(sig_fit, water_content, sig_s)

    # er_s_opt[0] = 0.754
    # er_f_opt[0] = 0.312
    # sig_s_opt[0] = 0.012

    er_s_squared = r_squared(er_s_opt, er_fit, water_content, er_s)
    er_f_squared = r_squared(er_f_opt, er_fit, water_content, er_f)
    sig_s_squared = r_squared(sig_s_opt, sig_fit, water_content, sig_s)
    test_water_content = np.linspace(0,100)
    print(f"R-squared er_s: {er_s_squared:0.2f}")
    print(f"Slope er_s: {er_s_opt}")
    plt.plot(water_content, er_s, 'r.', label = 'data')
    plt.plot(test_water_content, er_fit(test_water_content, *er_s_opt) ,label = 'fit')
    plt.legend()
    plt.title("er_s")
    plt.xlabel("Water Content [%]")
    plt.ylabel("er_s []")
    plt.ylim([0,80])
    plt.xlim([0,100])
    plt.show()
    print(f"R-squared er_f: {er_f_squared:0.2f}")
    print(f"Slope er_s: {er_f_opt}")
    plt.plot(water_content, er_f, 'r.', label = 'data')
    plt.plot(test_water_content, er_fit(test_water_content, *er_f_opt),label = 'fit')
    plt.xlabel("Water Content [%]")
    plt.ylabel("er_f []")
    plt.title("er_f")
    plt.ylim([0,80])
    plt.xlim([0,100])
    plt.show()
    print(f"R-squared sig_s: {sig_s_squared:0.2f}")
    print(f"Slope er_s: {sig_s_opt}")
    plt.plot(water_content , sig_s, 'r.',label = 'data')
    plt.plot(test_water_content, sig_fit(test_water_content, *sig_s_opt),label = 'fit')
    plt.title("sig_s")
    plt.ylim([0,8])
    plt.xlim([0,100])
    plt.xlabel("Water Content [%]")
    plt.ylabel("Sigma_s [S/m]")
    plt.show()

    data ={}
    data['er_s'] = {'m':er_s_opt[0],'r2':er_s_squared}
    data['er_f'] = {'m':er_f_opt[0],'r2':er_f_squared}
    data['sig_s'] = {'m':sig_s_opt[0],'r2':sig_s_squared}
    
    with open(os.path.join(folder, file_name) + '_slope.json','w') as json_file:
        json.dump(data, json_file, indent=4)


def fit_debye(output_folder, file_name, fmin, fmax, tau,reference):
    fmin = float(fmin)
    fmax = float(fmax)
    tau = float(tau)
    database_path = "C:\\Users\\brendon.besler\\MicrowaveInversion\\debye\\dielectric_properties_itis_v3p0.xls"
    tissue_list = ["Muscle", "Skin", "Fat", "Bone(Cortical)",  "Blood","Bone(Cancellous)"]
    #f = np.logspace(np.log10(fmin), np.log10(fmax), num=1001)
    f = np.linspace(fmin, fmax, num=1001)
    # load spreadsheet
    wb = xlrd.open_workbook(database_path)
    # get first sheet
    sheet = wb.sheet_by_index(0)
    # get list of tissues
    tissues = sheet.col_values(0)
    
    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # delete old file or use new name
    if os.path.exists(os.path.join(output_folder, file_name + ".json")):
        delete_file = input("Json debye fit file exists. Delete it? [y/n]")
        if delete_file == 'y':
            print("Deleting file")
            os.remove(os.path.join(output_folder, file_name + ".json"))
        elif delete_file == 'n':
            print("Changing file name")
            file_name += '1'
        else:
            sys.exit("Invalid input... terminating")
    data = {}
    # go through all the tissues of interest
    for tissue in tissue_list:
        # find row of tissue
        row_indx = tissues.index(tissue)
        row = sheet.row_values(row_indx)

        # initialize cole-cole model from table
        cole_model = er_model.ColeCole(row, tissue, f)

        # initial guess at debye parameters
        x0 = [cole_model.ef, cole_model.er.real[0],cole_model.sig_s]
        # do least squares fit
        res2 = optimize.least_squares(er_model.Debye.debye_fit_accurate, x0, args=(f, cole_model.er))
        #res = leastsq(Debye.debye_fit_accurate, x0, args=(f, cole_model.er))

        #debye_model = Debye(res[0][0], res[0][1], 17.5e-12, res[0][2], f)
        
        debye_model = er_model.Debye(res2.x[0], res2.x[1], tau, res2.x[2], f)

        # values from David's paper (2-12 GHz)
        if tissue == 'Muscle':
            if reference:
                debye_model = er_model.Debye(25.40, 58.22, 17.5e-12, 0.664, f)#muscle
            water_frac = 0.741
        elif tissue == 'Skin':
            if reference:
                debye_model = er_model.Debye(18.51, 42.28, 17.5e-12, 0.679, f)#skin
            water_frac = 0.653
        elif tissue == 'Bone(Cancellous)':
            if reference:
                debye_model = er_model.Debye(6.7, 19.41, 17.5e-12, 0.331, f)#bone(canc)
            water_frac = 0.23
        elif tissue == 'Bone(Cortical)':
            if reference:
                debye_model = er_model.Debye(4.73, 11.91, 17.5e-12, 0.131, f)#bone(cort)
            water_frac = 0.15
        elif tissue == 'Fat':
            if reference:
                debye_model = er_model.Debye(5.97, 11.55, 17.5e-12, 0.076, f)#fat(inf)
            water_frac = 0.212
        elif tissue == 'Blood':
            if reference:
                debye_model = er_model.Debye(24.72, 64.22, 17.5e-12, 1.24, f)
            water_frac = 0.8
        else:
            print(f"Tissue: {tissue} not found")

        error = np.sum(np.abs(debye_model.er - cole_model.er)**2) / debye_model.f.size

        # format dictionary to right out

        data[tissue] = {
            "er_inf":debye_model.er_f, 
            "er_s":debye_model.er_s, 
            "tau":tau, "sig_s":debye_model.sigma, 
            "water":water_frac, 
            "error":error}


        print(tissue)
        print(f"Error = {error:0.2f}")
        print(f"er_f: {debye_model.er_f:0.2f}")
        print(f"er_s: {debye_model.er_s:0.2f}")
        print(f"sig_s: {debye_model.sigma:0.2f}")

        plt.plot(debye_model.f, debye_model.er.real, label ="Debye")
        plt.plot(cole_model.f, cole_model.er.real, label = "Cole-Cole")
        #plt.plot(debye_model_source.f, debye_model_source.er.real, label ="David Debye")
        plt.ylabel("er'")
        plt.xlabel('Frequency [Hz]')
        plt.ylim([0,60])
        plt.title(tissue)
        plt.legend()
        plt.grid()
        plt.show()

        plt.plot(debye_model.f, debye_model.er.imag, label ="Debye")
        plt.plot(cole_model.f, cole_model.er.imag, label = "Cole-Cole")
        #plt.plot(debye_model_source.f, debye_model_source.er.imag, label ="David Debye")
        plt.legend()
        plt.grid()
        plt.ylim([0,-50])
        plt.xlabel('Frequency [Hz]')
        plt.title(tissue)
        plt.ylabel("er''")
        plt.show()
    # write dictionary to .json
    with open(os.path.join(output_folder, file_name + ".json"),'a') as file:
        json.dump(data,file,indent=4)



def main():
    description ='''Fit debye model to measured Cole-Cole data and apply hydration models

    Example usage:
    debye_fit results test_debye 1e9 10e9 1e-9

    Calls the inverse function which:
        - reads in tissue data from database
        - fits debye model to database data
        - fits debye model vs water
        - saves debye parameters and slopes
    
    '''
    # Setup argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="debye_fit",
        description=description
    )
    parser.add_argument('output_folder', help='Folder to save data to')
    parser.add_argument("file_name", help="Name for file (no extension)")
    parser.add_argument('fmin', help='Minimum frequency')
    parser.add_argument('fmax', help='Maximum frequency')
    parser.add_argument('tau', help='Time constant')
    parser.add_argument('-r','--reference', help="Use reference values?", required=False,action="store_true")
    

    # Parse args and display
    args = parser.parse_args()

    # Run program
    fit_debye(**vars(args))

    # Run other program
    fit_water(args.output_folder, args.file_name)

    

if __name__ == "__main__":
    main()