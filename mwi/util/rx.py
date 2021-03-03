import numpy as np
import math
from matplotlib import pyplot as plt

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

    def calc_rx_angle(self):
        """Calculate angle for rx positions"""
        return self.calc_angle(self.nrx)
    
    def calc_tx_angle(self):
        """Calculate angle for tx positions"""
        return self.calc_angle(self.ntx)
    
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

    def calc_pos_discrete(self,N,dx,dy):
        """Places rx/tx at nearest grid line/discretization. Note: due to floating point precision mid points may not be rounded correctly (e.g. sin(pi/3) + 1 = 1.5 = 1.499996 should round up)

        Args:
            - N (int): number of poitns
            - dx (float): discretization in x
            - dy (float): discretization in y
        
        Outputs:
            - np.ndarray: 2 x N of discrete (x,y) coordinates for each position
        """
        # Do placement calculation to nearest discretization
        x = np.zeros(N)
        y = np.zeros(N)
        
        
        """ TODO With nrx = 48, ntx = 24, r = 0.075, dx = dy = 0.01 there is a 4 position that are different... result is 10% worse...
        pos = self.calc_pos(N)
        x = pos[0, :]
        y = pos[1, :]

        x = np.round(x / dx) * dx
        y = np.round(y / dy) * dy
        """
        # 0.99 helps with accurate rounding    
        for i in range(N):
            angle = i*2*math.pi/N
            x[i] = 0.99*self.r*math.cos(angle) + self.x0 
            x[i] = np.round(x[i]/dx)*dx
            y[i] = 0.99*self.r*math.sin(angle) + self.y0
            y[i] = np.round(y[i]/dy)*dy
        return np.array([x,y])

    def calc_tx_discrete(self,dx,dy):
        """calculate Tx locations rounded to nearest discretization step (dx,dy)"""
        return self.calc_discrete(self.ntx,dx,dy)
    
    def calc_rx_round(self,dx,dy):
        """calculate Rx locations rounded to nearest discretization step (dx,dy)"""
        return self.calc_round(self.nrx,dx,dy)
    
    def translate(self, dx, dy):
        """Translate measurements surface by (dx,dy)"""
        self.x0 += dx
        self.y0 += dy

    def is_too_close(self, angle):
        """Returnds 2D boolean array with True if rx and tx are too close (within angle). Negative angle is ignored (returns all false).

        Args:
            -angle (float): angle separation (in radian) to test

        Outputs:
            - too_close (np.ndarray, dtype = bool): 2D array of bools (nrx by ntx). Returns TRUE if rx/tx are within angle, returns FALSE else. 
        """
        if angle >=0:
            rx_angle = self.calc_rx_angle()
            tx_angle = self.calc_tx_angle()

            rx_angle2 = np.tile(rx_angle, (self.ntx, 1))
            tx_angle2 = np.tile(tx_angle, (self.nrx, 1))

            # calculate difference between angles and normalize to pi
            dif =  (rx_angle2.T - tx_angle2)
            dif = dif % (2*math.pi)
            dif = (dif + 2*math.pi) % (2*math.pi)
            idx = (dif > math.pi)
            dif[idx] += -2*math.pi
            # find which rx to tx angles are greater than threshold
            too_close = (np.absolute(dif) - angle/2 <= -0.0001)
        else:
            too_close = np.zeros((self.nrx, self.ntx), dtype = bool)

        return too_close

