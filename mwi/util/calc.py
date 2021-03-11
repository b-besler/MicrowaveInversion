import numpy as np
from scipy import special
from scipy import linalg
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
    angle_idx = model.rx.is_too_close(model.image_domain.ex_angle)

    nx = model.image_domain.x_cell.size
    ny = model.image_domain.y_cell.size

    data_operator = np.zeros((model.ntx* (model.nrx - np.sum(angle_idx, axis=0)[0]) * model.nf, ny * nx), dtype=complex)

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

def L_curve(rx_data, data_operator, n_gamma):
    """ Do L-Curve -- more info in "Rank-Deficient and Discrete Ill-Posed Problems" or "The L-Curve and its use in the nuermical treatment of inverse problems" (P. C. Hansen)
    L-Curve is a log-log plot of the solutin norm vs the residual norms for different regularization parameter. Currently uses linear Tikhonov regularization.
    
    Args:
        - rx_data (np.ndarray): vector of measured scattered data (frequency domain)
        - data_operator (np.ndarray): data operator matrix
        - n_gamma (int): number of points to consider
    
    Outputs:
        - res_norm (np.ndarray): vector of residual norms (L2)
        - soln_norm (np.ndarray): vector of corresponding solution norms (L2)
        - 
    """

    # do singular value decomp of M matrix (hermitian)
    (_,s,_) = linalg.svd(form_hermitian(data_operator))

    # define gamma values based on maximum singular value
    gamma = np.amax(s)*np.logspace(-8,0,n_gamma) 

    # Tikhonov Regularization # TODO add other regularization schemes (divergence [Lo Vetri "Enhanced DBIM"])
    R = np.identity(data_operator.shape[1])

    # initialize residual norm and solution norm vectors
    res_norm = np.zeros(n_gamma)
    soln_norm = np.zeros(n_gamma)

    for i in range(n_gamma):
        soln = solve_regularized(rx_data, data_operator, gamma[i], R)
        res_norm[i] = linalg.norm(rx_data - data_operator @ soln, 2)
        soln_norm[i] = linalg.norm(soln, 2)

    return (res_norm, soln_norm, gamma)

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

def form_hermitian(A):
    """ Forms hermitian matrix by multiplying by complex conjguate tranpose (A_h = A'A)
    Args:
        - A (np.ndarray): m x n matrix
    Output:
        - n x n hermitian matrix
    """

    # TODO is there a faster/better way to do this using vector multiplication
    assert A.ndim == 2, 'A matrix must be m x n to form hermitian'
    
    output = np.conjugate(A.T) @ A
    
    return output

def solve_regularized(b, A, gamma, R):
    """Solve min||b-Ax||_2 + gamma*||xR||_2 - see Wang and Chew 1989
        Equivalent to solving x = [A'A + gamma R'R]^-1 A' b

    Args:
        - b (np.ndarray): array with expected values (i.e. measured scattered field)
        - A (np.ndarray): data matrix
        - gamma (float): regularization parameter
        - R (np.ndarray): regularization matrix (identity for normal Tikhonov)

    Output:
        - x (np.ndarray): solution to about equation
    """

    A_reg = form_hermitian(A) + gamma * form_hermitian(R)

    condition_num = np.linalg.cond(A_reg)

    if condition_num > 1e14:
        print("Warning: Regularized matrix is poorly conditioned, results may not be accurate")
    
    return linalg.inv(A_reg) @ np.conjugate(A.T) @ b