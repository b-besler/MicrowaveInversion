
# Function related to Method of Moments solver
# Bases on Richmond, Harrington, 

import numpy as np
from scipy import special
from scipy import signal
from matplotlib import pyplot as plt
import time

import mwi.util.constants as constants

def sim_homogeneous_background(background, obj, current):
    """Calculates the "measured" scattered fields at receivers from transmitters

    """
    inc_fields = calc_homogeneous_fields(background, current)

    (total_fields,_) = calc_total_fields(inc_fields,obj)

    scat_rx = calc_rx_scatter(obj, total_fields)


    return (inc_fields, total_fields, scat_rx)

def calc_homogeneous_fields(model, I):
    """ Calculates the homogeneous field distriubtion due to point source. Microwave Imaging eqn 2.4.26.
    
    Args:
        - model (class): model object with geomety and background medium information
        - I (complex): complex current of source
    
    Output:
        - nrx x ny x nx array with fields of x,y due to point sources
    """

    return (1j *  2 * np.pi * model.freq * constants.MU0 * I) * homogeneous_greens(model)

def homogeneous_greens(model):
    """Calculate homogeneous green's function due to transmitter at rx at all points x and y. 
    Assumes time harmonic is exp(+jwt). Microwave imaging eqn 3.3.9.

    Args:
        - x (np.ndarray): array of x coordinates (cell centered)
        - y (np.ndarray): array of y coordinates (cell centered)
        - rx (np.ndarray): 2 x nrx array of source coordinates
        - er (complex): complex permittivity of background medium
        - f (float): frequency

    Output:
        - returns matrix of size ny x nx x nrx
    """
    x = model.image_domain.x_cell
    y = model.image_domain.y_cell
    rx = model.rx.calc_rx_discrete(model.dx, model.dy)

    # calculate distances
    rho = np.zeros((model.nrx, y.size, x.size))
    for i in range(model.nrx):
        rho[i,:,:] = np.sqrt(np.add.outer((y - rx[1,i])**2, (x - rx[0,i])**2))

    # if rho == 0 avoid NaN. Note: not accurate.
    indx = np.argwhere(rho == 0.0)
    if not(indx.size == 0):
        rho[indx] = rho[indx[0]+1]

    # Calculate fields in background medium
    greens = -1/(1j*4)*special.hankel2(0, rho*model.k)

    return greens

def homogeneous_solution(rx, tx, er, f, I):
    """Calculate homogeneous solution due from tx to rx. Note monostatic solution is not accurate.
    

    -Args:
        - rx (np.ndarray): 2 x nrx array of rx coordinates
        - tx (np.ndarray): 2 x ntx array of tx coordinates
        - er (complex): complex permittivity of background (single frequency)
        - f (float): frequency
        - I (complex): complex components of source current
    """

    rho = np.sqrt(np.add.outer(tx[0], -rx[0])**2 + np.add.outer(tx[1], - rx[1])**2)
    # deal with discontinuity at rx=tx (note not accurate)
    indx = np.argwhere(rho == 0.0)
    rho[indx] = rho[indx[0]+1]

    k = 2*np.pi*f/constants.C0 * np.sqrt(er)

    field = 1j*2*np.pi*f*constants.MU0*-1j/4*I*special.hankel2(0, rho*k)

    return field

def calc_total_fields(inc_fields, model):
    """Calculate the total fields in domain. Uses Richmond's Method (1965), eqn 14-16

    Args:
        - inc_fields (np.ndarray): ntx x ny x nx array with incident fields due to transmitter i over (x,y)
        - model (class): model object with information about object geometry and material
    Outputs:
        - total field for each transmitter (ntx x ny x nx)
        - inc_fields to total fields transform (ntx x (ny x nx) x (ny x nx))
    """
    tx = model.rx.calc_tx_discrete(model.dx, model.dy)
    x = model.image_domain.x_cell
    y = model.image_domain.y_cell

    # Initialize arrays
    fields = np.zeros((tx.shape[1], y.size,x.size,), dtype = np.complex_)
    C = np.zeros((x.size*y.size, x.size*y.size), dtype = np.complex_)

    # iterate over row in source image
    idx = 0
    for j in range(y.size):
        # iterate over pixel in source image row
        for k in range(x.size):
            # calculate distance from source point to observation points
            rho_matrix = np.sqrt(np.add.outer((y[j] - y)**2, (x[k] - x)**2))
            C_matrix = -model.k**2*hankel_integral_cell(model.k, model.a, rho_matrix)*model.get_image('contrast')
            # find rho = 0 case and do special evaluation
            indx = (rho_matrix == 0)
            if not (indx.size == 0):
                C_matrix[indx] += 1 
            # Add C matrix for this pixel to total matrix
            C[idx,:] = np.ravel(C_matrix)
            idx += 1
    # TODO: compute inverse using conjugate gradient method
    C_inv = np.linalg.inv(C.T)
    
    # iterate over ntx
    for i in range(tx.shape[1]):
        # solve for total field and make into 2D image
        field_vec = inc_fields[i,:,:].flatten().T @ C_inv
        fields[i,:,:] = np.reshape(field_vec,(y.size, x.size))
    
    return (fields,C_inv)

def calc_rx_scatter(model, fields):
    """Calculate scattered fields at receivers (points outside image domain)

    Args:
        - model (class): object with geometry, materials, and receivers
        - fields (np.ndarray): ntx x ny x nx array with total fields due to each transmitter
    """
    scat_rx = np.zeros((model.nrx, model.ntx), dtype = np.complex_)

    rx = model.rx.calc_rx_discrete(model.dx, model.dy)
    y = model.image_domain.y_cell
    x = model.image_domain.x_cell

    # iterate over receiver locations
    for i in range(model.nrx):
        # iterate over transmitter (fields are a function of transmitter)
        for j in range(model.ntx):
            # calculate distance between image domain pixels and receiver
            rho = np.sqrt(np.add.outer((rx[1,i] - y)**2, (rx[0,i] - x)**2))
            # calculate hankel integral (note rho =0 will give NaN result at that point)
            hank = hankel_integral_cell(model.k, model.a, rho)
            # calculate scattered field as sum of contribution of pixels
            scat_rx[i,j] =  model.k**2*np.sum(model.get_image('contrast') * np.squeeze(fields[j,:,:]) *hank)

    return scat_rx

def calc_scatter_field(model, total_field):
    """Calculate scattered field from total field
    """
    x = model.image_domain.x_cell
    y = model.image_domain.y_cell
    scat_field = np.zeros((model.ntx, y.size, x.size), dtype = np.complex_)
    scat_field2 = np.zeros((model.ntx, y.size, x.size), dtype = np.complex_)

    # start = time.time()
    # # iterate over transmitter (fields are a function of transmitter)
    # for L in range(model.ntx):
    #     current = (model.get_image('comp_er') -1) * np.squeeze(total_field[L,:,:])
    #     # iterate over y pixel
    #     for i in range(y.size):
    #         # iterate over x pixel
    #         for j in range(x.size):
    #             # calculate distance from source pixel to observation pixel
    #             rho = np.sqrt(np.add.outer((y[i] - y)**2, (x[j] - x)**2))
    #             # calculate hankel integral, if rho = 0 do special evaluation
    #             hank = (1j*np.pi*model.k*model.a/2) * special.jv(1, model.k*model.a) * special.hankel2(0, model.k*rho)
    #             # find rho = 0 elements and do special evaluation
    #             indx = (np.argwhere(np.isnan(hank)))
    #             if not (indx.size == 0):
    #                 hank[indx[0,0],indx[0,1]] = (1j/2)*(np.pi*model.k*model.a*special.hankel2(1,model.k*model.a) - 2*1j)
    #                 # scattered fields are the sum of contributions from each pixel
    #             scat_field[L,i,j] = -np.sum(current *hank)
    # print(f"Time elapsed for loops: {time.time()-start:0.2f}")

   
    xx, yy = np.meshgrid(x,y)
    rho = np.sqrt((xx-model.dx/2)**2 + (yy-model.dy/2)**2)
    hankel_image = hankel_integral_cell(model.k, model.a, rho)
    contrast = model.get_image('contrast')
    for L in range(model.ntx):
        current = contrast* np.squeeze(total_field[L,:,:])
        scat_field2[L,:,:] = model.k**2*signal.fftconvolve(current, hankel_image, mode='same')

    return scat_field2

def hankel_integral_cell(k,a,rho):
    """Calculate the integral of hankel function inside and outide of cell over cell (Richmond 1965, eqn 12)
    Integral of -1j/4 H^2_0(k rho) i.e. 2D homogeneous green's function
    Args:
        - k (complex): wavenumber
        - a (float): radius of circle of equivalent area to cell
        - rho (np.ndarray): array of distances from center of this cell to observation point
    Output:
        - hank_int (np.ndarray): array of same size as rho with evaluated integral"""
    hank_int = -(1j*np.pi*a/(2*k)) * special.jv(1, k*a) * special.hankel2(0, k*rho)
    indx = (np.argwhere(np.isclose(rho,0)))
    if not (indx.size == 0):
        hank_int[indx[:,0],indx[:,1]] = -(1j/2)*(np.pi*a/k*special.hankel2(1,k*a) - 2*1j/k**2)
    return hank_int


def form_greens_operator(model, greens, field):
    """Forms the data operator matrix used for inverse. M matrix in Moghaddam + Chew 1992.
    Args:
        - model (class): class holding model information including imaging domain
        - greens (np.ndarray): matrix holding greens function integrals
        - field (np.ndarray): matrix holding field information
    Output:
        - data_operator (np.ndarray): matrix that relates contrast to measured scattered fields (size: [ntx x nrx x nf] x  [ny x nx])
    """
    # find indices of angles to exlude
    angle_idx = model.rx.is_too_close(model.image_domain.ex_angle)

    nx = model.image_domain.x_cell.size
    ny = model.image_domain.y_cell.size

    data_operator = np.zeros((model.ntx* (model.nrx - np.sum(angle_idx, axis=0)[0]), ny * nx), dtype=complex)

    idx = 0
    for i in range(model.nrx):
        for j in range(model.ntx):
            # skip rx/tx that are too close
            if (not angle_idx[i,j]):
                    # Combine wavenumber of background (k2 = w2 eps mu), total field and greens function
                    image = model.k**2 * field[j, : :] * greens[i,:,:]
                    # Flatten 2D array into 1D array for each nrx/ntx
                    data_operator[idx, :] = image.flatten()
                    idx += 1
    
    return data_operator


def greens_from_fields(model, fields, I):
    """Calculate the green's function of a point source in 2D region using the transmitted fields in the region
    
    """

    return -fields/ (1j*I*2*np.pi*model.freq*constants.MU0) * model.dx * model.dy

def calc_hankel_integral(model):
    """Calculate the response over each pixel in image to each receiver."""

    hank_int = np.zeros((model.nrx, model.image_ny, model.image_nx), dtype = np.complex_)

    # get points of iterest
    (x_coord, y_coord) = np.meshgrid(model.image_domain.x_cell, model.image_domain.y_cell)
    pos = model.rx_pos
    rx = pos[0,:]
    ry = pos[1,:]
    for i in range(model.nrx):
        # form rho matrix with distance from each point in image to rx
        rho = np.sqrt((rx[i] - x_coord)**2 + (ry[i] - y_coord)**2)
        # calculate integral using Richmond's method (Richmond, 1965)
        hank_int[i, :, :] = hankel_integral_cell(model.k,model.a,rho)#-1j/4 * 2*np.pi*model.a/model.k * special.jv(1, model.k * model.a) * special.hankel2(0, model.k * rho)
    
    return hank_int

def convolution(image, kernel):
    """2D Convolution between two images (image and kernel).
    """

    if not(image.shape == kernel.shape):
        raise ValueError("Image and Kernel must be same size")
    
    if not (image.ndim == 2) or not(kernel.ndim == 2):
        raise ValueError("Image and Kernel must be 2D arrays")

    image_f = np.fft.fft2(image)
    kernel_f = np.fft.fft2(kernel)

    product = image_f * kernel_f

    result = np.fft.ifft2(product)
    return result

def add_noise(signal, noise_level):
    """Add white noise to a signal. Noise_level is the noise amplitude defined as % of signal peak
    Args:
        - signal (np.ndarray): 1D array of complex signal
        - noise_level (np.ndarray): noise amplitude defined as % of signal peak
    Output:
        - signal corrupted with noise (1D array)
    """

    peak = np.max(np.abs(signal))
    alpha = np.random.normal(0,0.33,signal.size)
    beta = np.random.normal(0,0.33,signal.size)

    return signal + peak * noise_level * (alpha + 1j*beta)

