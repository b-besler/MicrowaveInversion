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
                    image = (constants.E0 * (2*np.pi*f)**2 * constants.MU0 * field[j,f_field_idx[k], : :] * hank_int[i, k, :, :])
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

def L_curve_knee(res, soln, gamma):
    """ Find the knee point (maximum curvature) of L-curve

    Args:
        - res (np.ndarray): residual norms for each gamma
        - soln (np.ndarray): solution norms for each gamma
        - gamma (np.ndarray): regularization parameter gamma

    Output:
        - gamma_max (float): gamma of maximum curvature
        - idx (int): index corresponding to gamma
    """

    kappa = curvature(res, soln, gamma, 5, 5)

    # any NAN replace with 0
    kappa = np.nan_to_num(kappa)

    idx = np.argmax(kappa)

    return (kappa, gamma[idx], idx)

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

def curvature(x, y, t, n, order):
    """ Calculate curvature of a parametric curve [x(t), y(t)]
        At each point t, a polynomial of order "order" is fit to the n points before and after it
        Derivaties are then aculated based on the polynomial fit

    Args:
        - x (np.ndarray): array of x(t) values
        - y (np.ndarray): array of y(t) values
        - t (np.ndarray): array of t values
        - n (int): number of points before and after to fit
        - order (int): polynomial order
    
    Output:
        - kappa (np.ndarray): curvature at each point t
    """                
    # Note: this method is robust but not the fastest. The curvature calculation can be complicated due to the shape of the L-curve (i.e. highly logarithmic)
    # other options - vector curvature or meneger. Sensitive to number of points/closeness
    if not(int(n) == n):
        raise ValueError("n must an integer")

    if not(int(order) == order):
        raise ValueError("order must be an integer")

    if 2*n < order:
        raise ValueError("Need more points to fit polynomial of order " + str(order))

    if not(x.size == y.size) or not(x.size == t.size):
        raise ValueError("x, y, t arrays must be the same size")

    N = t.size

    # initialize derivative arrays
    dx = np.zeros(x.size)
    dy = np.zeros(y.size)
    ddx = np.zeros(x.size)
    ddy = np.zeros(y.size)

    # iterate through each point
    for i in range(n, N-n):
        # select small section of curve
        x_spline = x[i-n : i+n]
        y_spline = y[i-n : i+n]
        t_spline = t[i-n : i+n]

        # fit polynomial x(t), do x'(t), x''(t)
        px = np.polyfit(t_spline, x_spline, order)
        dpx = np.polyder(px)
        ddpx = np.polyder(dpx)

        # fit polynomial y(t), do y'(t), y''(t)
        py = np.polyfit(t_spline, y_spline, order)
        dpy = np.polyder(py)
        ddpy = np.polyder(dpy)

        # evaluate derivaties
        dx_spline = np.polyval(dpx, t_spline)
        ddx_spline = np.polyval(ddpx, t_spline)
        dy_spline = np.polyval(dpy, t_spline)
        ddy_spline = np.polyval(ddpy, t_spline)

        # select derivatives at mid point of spline
        dx[i] = dx_spline[n]
        dy[i] = dy_spline[n]
        ddx[i] = ddx_spline[n]
        ddy[i] = ddy_spline[n]
    
    # assume derivates are constant at the beginning and end of curve
    dy[range(n)] = dy[n]
    dx[range(n)] = dx[n]
    ddy[range(n)] = ddy[n]
    ddx[range(n)] = ddx[n]

    dy[range(N-n+1, N)] = dy[N-n-1]
    dx[range(N-n+1, N)] = dx[N-n-1]
    ddy[range(N-n+1, N)] = ddy[N-n-1]
    ddx[range(N-n+1, N)] = ddx[N-n-1]

    # calculate curvature, divide throws an error because some of the values are really small (1e-21)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        kappa = (dx * ddy - ddx * dy) / (dx**2 + dy**2)**1.5

    return kappa 

def select_data(model, rx_data, freq):
    """Select which rx data to include. Uses domain exclusion angle to decide which rx/tx to kick out data. Uses domain frequency to choose frequency components

    Args:
        - model (class): class with image domain information
        - rx_data (np.ndarray): matrix with rx data
        - freq (np.ndarray): array with frequencies
    Output:
        - returns rx_data array with proper rx/tx combinations and frequency points
    """

    angle_idx = model.rx.is_too_close(model.image_domain.ex_angle)

    (_, f_idx) = find_nearest(freq, model.freq)

    rx_out = np.zeros((model.nrx - np.sum(angle_idx, axis=0)[0]) * model.ntx * f_idx.size, dtype = np.complex_)

    idx = 0
    for i in range(model.nrx):
        for j in range(model.ntx):
            if not(angle_idx[i,j]):
                rx_out[idx:idx+f_idx.size] = rx_data[i,j,f_idx]
                idx += f_idx.size

    return rx_out

def menger_curvature(x, y, t):
    """ Calculate the menger curvature (three-point curvature).
    """
    N = t.size
    kappa = np.zeros(N)
    for i in range(1, N-1):
        # A,B,C are points in R3
        A = np.array([x[i-1], y[i-1], t[i-1]])
        B = np.array([x[i], y[i], t[i]])
        C = np.array([x[i+1], y[i+1], t[i+1]])

        # get displacement vectors
        AB = B - A
        AC = C - A
        BC = C - B

        # calculate area using cross product
        area = np.linalg.norm(np.cross(AB, AC))/2

        # calculate menger curvature
        kappa[i] = 4*area/(np.linalg.norm(AB) * np.linalg.norm(AC) * np.linalg.norm(BC))
    
    kappa[0] = kappa[1]
    kappa[N-1] = kappa[N-2]
    return kappa

def residuals_percent(actual, calculated):
    """ Calculate residuals in percent (normalized to expected data) and the root mean square error in percent.

    Args:
        - actual (np.ndarray): expected values
        - calculated (np.ndarray): measured or calculated data
    Outputs:
        - rmse (float): mean square error in percent
    """

    if np.any(actual == 0.0):
        raise ValueError("Actual data cannot equal 0 for any element")

    residuals = (calculated - actual)/actual * 100

    rmse = np.sqrt(np.sum(np.absolute(residuals**2))/residuals.size)

    return (residuals,rmse)