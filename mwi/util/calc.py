import numpy as np
from scipy import special
import mwi.util.constants as constants

def hankel_integral(model):
    """Calculates the back propagation (homogenous 2D Green's Function) from any point in image domain to receivers
        Approximates square basis as circle of equal area to get close form (Richmond, 1965)

    Args:
        - model (class): model class with rx locations and image domain information

    Output:
        - hankel_int (np.ndarray): hankel integral from each image pixel to each receiver (size nrx x nf x ny x nx)
    """
    # initialize matrix
    hank_int = np.zeros((model.nrx, model.nf, model.image_domain.y_cell.size, model.image_domain.x_cell.size), dtype = np.complex)

    # calculate square area and equivalent circle radius
    delta_S = model.image_domain.dx * model.image_domain.dy
    a = np.sqrt(delta_S/np.pi)

    (rx_x, rx_y) = model.rx.calc_rx_discrete(model.image_domain.dx, model.image_domain.dy)

    for i in range(model.nrx):
        for j in range(model.nf):
            f = model.freq[j]
            # wave number
            kj = 2*np.pi*f/constants.C0
            for L in range(model.image_domain.y_cell.size):
                for k in range(model.image_domain.x_cell.size):
                    rho = np.sqrt((rx_x[i] - model.image_domain.x_cell[k])**2 + (rx_y[i] - model.image_domain.y_cell[L])**2)
                    hank_int[i,j,L,k] = -1j/4 * (2*np.pi*a)/kj * special.jv(1, kj*a) * special.hankel2(0, kj*rho)
    return hank_int

def form_data_operator(model, hank_int, field, field_freq):
    """Forms the data operator matrix used for inverse. M matrix in Moghaddam + Chew 1992.
    Args:
        - model (class): class holding model information including imaging domain
        - hankel_int (np.ndarray): matrix holding hankel function integrals (i.e. homogenous 2D Green's function)
        - field (np.ndarray): matrix holding field information
        - field_freq (np.ndarray): vector with frequencies corresponding to field data
    Output:
        - data_operator (np.ndarray): matrix that relates contrast to measured scattered fields (size: [ntx x nrx x nf] x  [ny x nx])
    """
    # find indices of angles to exlude
    angle_idx = np.zeros((4,4), dtype = bool)#model.rx.is_too_close(model.image_domain.ex_angle)

    nx = model.image_domain.x_cell.size
    ny = model.image_domain.y_cell.size

    data_operator = np.zeros((model.ntx* (model.nrx - np.sum(angle_idx, axis=0)[0]) * model.nf, ny * nx), dtype=complex)
    print(data_operator.shape)
    (f_nearest, f_field_idx) = find_nearest(field_freq, model.freq)

    idx = 0
    for i in range(model.nrx):
        for j in range(model.ntx):
            # skip rx/tx that are too close
            if (not angle_idx[i,j]):
                for k in range(model.nf):
                    f = model.freq[k]
                    image = (constants.E0 * (2*np.pi*f)**2 * constants.MU0 * field[j,f_field_idx[k], : :] * hank_int[j, k, :, :])
                    data_operator[idx, :] = image.flatten()
                    idx += 1
    
    return data_operator

def find_nearest(array,value):
    """ findes closest element in array to value, returns the value it found the index. Array must be sorted
    Args:
        - array (np.ndarray): sorted array of floats
        - value (np.ndarray): array of floats to find nearest value
    Outputs:
        - nearest_val: closest value
        - idx: indices corresponding to nearest values
    """
    idx = np.searchsorted(array, value, side="left")
    idx = idx - (np.abs(value - array[idx-1]) < np.abs(value - array[idx]))
    return (array[idx],idx)