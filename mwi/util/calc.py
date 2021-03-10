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
                    rho = np.sqrt((rx_x[i] - model.image_domain.x_cell[k])**2 + (rx_y[i] - model.image_domain.y_cell[L]))
                    hank_int[i,j,L,k] = -1j/4 * (2*np.pi*a)/kj * special.jv(1, kj*a) * special.hankel2(0, kj*rho)
    return hank_int


