import numpy as np
import math

class MeasurementSurface():
    """Class for circular measurement surface defined by center (x0,y0), radius (r), and number of receivers/transmitters (rx/tx)
    """

    def __init__(self, config):
        """Initialize measurement surface class with configuration information.
        
        Args:
            - config (dict): measurement configuration file (.json) (see examples/measurement_surface.json)
        """

        self.x0 = config["x0"]
        self.y0 = config["y0"]
        self.nrx = config["nr"]
        self.ntx = config["nt"]
        self.r = config["r"]
    
    def calc_angle(self, num):
        """Calculate angle of rx/tx placed equidistant around circle starting at +x-axis
        """
        if num < 1:
            raise ValueError("Number of rx or tx must be greater than 0")

        n = np.linspace(0, num - 1, num = num)
        return n*2*math.pi/num
    
    def calc_pos(self, n):
        """Calculate (x,y) position of rx/tx placed equidistant around circle starting at +x-axis

        Args:
            - n (int): number of positions (usually nrx or ntx)

        Outputs:
            - pos (np.ndarray): 2 x n array with x, y coordinates (float)
        """
        x = self.r * np.cos(self.calc_angle(n)) + self.x0
        y = self.r * np.sin(self.calc_angle(n)) + self.y0

        return np.array([x,y])

    def calc_rx(self):
        """ Calculates placement of Rx, placed equidistant around center """
        return self.calc_pos(self.nrx)
               
    def calc_tx(self):
        """Calculates placement of Tx placed equidistant around center """
        return self.calc_pos(self.ntx)