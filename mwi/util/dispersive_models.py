# Debye and Cole-Cole model classes
import numpy as np
import json

import mwi.util.constants as constants

def assign_properties(objects, materials, f):
    """Assign complex permittivity values to object

    Args:
        - object (dict): dictionary of objects with er and sig and material name
        - materials (dict): definition of materials
        - f (float): frequency to evaluate properties
    """
    # iterate through objects
    for obj in objects:
        # grab object material name
        material_name = obj['name']
        # make sure material name is in matieral config
        if material_name in materials:
            print(f"Updating {material_name} properties.")
            # create debye model from debye parameters
            debye_model = Debye(
                materials[material_name]['er_inf'], 
                materials[material_name]['er_s'], 
                materials[material_name]['tau'], 
                materials[material_name]['sig_s'],
                f[0]
            )
            # assign er/sig at f to object
            temp = {'er':debye_model.er.real, 'sig':debye_model.sig}
            obj.update(temp)
        else:
            print(f"Unknown material {material_name}. Skipping...")

def read_materials(file):
    """Read in .json file to dict"""
    with open(file,'r') as json_file:
        materials = json.load(json_file)
    return materials

class Debye:

    def __init__(self, er_inf, er_s, tau, sigma, freq = np.logspace(1, 10, 101)):
        """Class for Debye dielectric properties model

        Args:
            - er_s (float): static permittivity
            - er_inf (float): optical permittivity
            - tau (float): relaxation constant
            - sigma (float): static conductivity
        """
        self.er_s = er_s
        self.er_f = er_inf
        self.tau = tau
        self.sigma = sigma
        self.f = freq


    @property
    def delta_er(self):
        return self.er_s - self.er_f

    @property
    def er(self):
        """Complex permittivity calculated via Debye model
        """
        return self.er_f + (self.er_s - self.er_f) / (1 + 1j * 2*np.pi * self.f * self.tau) + self.sigma / (1j * 2*np.pi * self.f * constants.E0)
    
    @property
    def sig(self):
        return np.abs(self.er.imag*(2*np.pi*self.f)*constants.E0)

    @staticmethod
    def debye_fit(p, f, y):
        """Function used for least square minimization of Debye model to measured data (Cole-Cole model)
        Args:
            - p (np.ndarray): array of minimization variables ([er_f, er_s, sigma]). tau is assumed constant
            - f (np.ndarray): array of frequency points
            - y (np.ndarray): array of measured complex permittivity (i.e. Gabriel data)

        Outputs:
            - returns the magnitude of the residuals (to be minimized via least squares)

        """

        er_f, er_s, sigma_s = p
        debye_model = Debye(er_f, er_s, 17.5e-12, sigma_s, f)
        return np.abs(debye_model.er - y)
    
    @staticmethod
    def debye_fit_accurate(p, f, y):
        """Error function to fit debye model to cole-cole model. See Lazebnik 2007.
        """
        er_f, er_s, sigma_s = p
        debye_model = Debye(er_f, er_s, 17.5e-12, sigma_s, f)
        real_er_median = np.median(debye_model.er.real)
        imag_er_median = np.median(debye_model.er.imag)
        error = ((y.real - debye_model.er.real)**2/real_er_median + (y.imag - debye_model.er.imag)**2/imag_er_median)/f.size
        return error

class ColeCole:

    def __init__(self, row, tissue, freq = np.logspace(1, 10, 101)):
        """Class to initalize cole-cole model from Gabriel database data. Uses the row of the IT'IS database excel data (DOI: 10.13099/VIP21000-03-0)

        Args:
            - row: excel spreadsheet row for tissue
            - tissue (string): tissue name
            - freq (np.ndarray, dtype = float): array of frequency points

        """
        self.tissue = tissue
        self.ef = row[1]
        self.del1 = row[2]
        tau1 = row[3]*1e-12
        alf1 = row[4]
        del2 = row[5]
        tau2 = row[6]*1e-9
        alf2 = row[7]
        self.sig_s = row[8]
        del3 = row[9]
        tau3 = row[10]*1e-6
        alf3 = row[11]
        del4 = row[12]
        tau4 = row[13]*1e-3
        alf4 = row[14]

        self.f = freq
        w = 2*np.pi*self.f
        self.er = self.ef\
                 + self.del1 / (1 + np.power(1j * w * tau1, 1- alf1))\
                 + del2 / (1 + np.power(1j * w * tau2, 1-alf2))\
                 + del3 / (1 + np.power(1j * w * tau3, 1-alf3))\
                 + del4 / (1 + np.power(1j * w * tau4, 1-alf4))\
                 + self.sig_s / (1j * w * 8.85e-12)

        self.sig = -self.er.imag*w*8.85e-12