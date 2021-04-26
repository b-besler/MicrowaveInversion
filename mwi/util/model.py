import numpy as np
import math
from matplotlib import pyplot as plt
import mwi.util.constants as constants

class Model():
    """Class for simulation model: permittivity and conductivity images, measurement surface, and imaging domain

    
    Known issues:
        -placing object at mid point between discretizations can cause errors due to float precision.
    """

    def __init__(self, config, rx_surface, image_domain, alt_grid):
        """Initialize class using configuration data, measurement surface, and imaging domain vector.

        Args:
            - config (dict): model configuration settings (see model_config.json in example)
            - rx_surface (class): class holding rx information
            - image_domain (class): class holding image domain information
        """
        # assign configuration data to class
        self.dx = config["dx"]
        self.dy = config["dy"]
        self.x1 = config["x1"]
        self.x2 = config["x2"]
        self.y1 = config["y1"]
        self.y2 = config["y2"]
        self.name = config["name"]

        # assign image domain
        self.image_domain = image_domain
        # assign alternative grid (use to avoid inverse crime)
        self.alt_grid = alt_grid

        # assign measurement surface
        self.rx = rx_surface

        # create 2D image using configuration data
        self.er = self.create_image(config, 'er')
        self.sig = self.create_image(config, 'sig')

        # keep track of some useful information
        self.er_b = config["objects"][0]["er"]


    #properties
    @property
    def x_size(self):
        return self.x2 - self.x1
    @property
    def y_size(self):
        return self.y2 - self.y1
    @property
    def x(self):
        return np.linspace(self.x1, self.x2, num = int(round(self.x_size/self.dx,1)) +1)
    @property
    def y(self):
        return np.linspace(self.y1, self.y2, num = int(round(self.y_size/self.dy,1)) +1)
    @property
    def x_cell(self):
        return np.linspace(self.x1 + self.dx/2, self.x2 - self.dx/2, num = int(round(self.x_size/self.dx, 1)))
    @property
    def y_cell(self):
        return np.linspace(self.y1 + self.dy/2, self.y2 - self.dy/2, num = int(round(self.y_size/self.dy, 1)))
    @property
    def nrx(self):
        return self.rx.nrx
    @property
    def ntx(self):
        return self.rx.ntx
    @property
    def freq(self):
        return self.image_domain.freq
    @property
    def nf(self):
        return self.image_domain.freq.size
    @property
    def rx_pos(self):
        return self.rx.calc_rx_discrete(self.image_domain.dx, self.image_domain.dy)
    @property
    def tx_pos(self):
        return self.rx.calc_tx_discrete(self.image_domain.dx, self.image_domain.dy)
    @property
    def a(self):
        """Radius of circle with area equal to area of discretization rectangle"""
        return np.sqrt(self.dx * self.dy / np.pi)
    @property
    def k(self):
        """Wavenumber in background medium at reconstruction frequencys"""
        return 2*np.pi*self.freq * np.sqrt(self.er_b) / constants.C0
    @property
    def image_area(self):
        return self.image_domain.area
    @property
    def image_nx(self):
        return self.image_domain.x_cell.size
    @property
    def image_ny(self):
        return self.image_domain.y_cell.size
    

    def create_image(self, config, prop):
        """Create 2D numpy image of property using objects in configuration

        Args:
            - config (dict): model configuration file
            - prop (str): string with property to create image of (i.e. 'er' or 'sig')

        Outputs:
            - image (np.ndarray): 2D array with objects written to file
        """
        # initialize image using background
        image = np.ones((self.y_cell.size, self.x_cell.size)) * config["objects"][0][prop]

        #TODO add smooth sine object
        for obj in config["objects"]:
            if obj["type"] == "background":
                pass               
            elif obj["type"] == "circle":
                # update image with circle
                self.add_circle(obj, image, obj[prop])
            elif obj["type"] == "ellipse":
                # update image with ellipse
                self.add_ellipse(obj, image, obj[prop])
            elif obj["type"] == "sinlike":
                self.add_sinlike(obj, image, obj[prop])
            elif obj["type"] == "square":
                self.add_square(obj, image, obj[prop])
            else:
                print("Unsupported object type: " + obj["type"] + ". Not added to image...")

        return image
    
    def add_circle(self, obj, image, value):
        """Create image of a circle with specified value.

        Args:
            - obj (dict): model configuration for object (contains (x0,y0), r, er, sig)
            - image (np.ndarray): 2D array to be written to (note: overwrites contents when adding circle contents)
            - value (float): value to be written for circle
        """ 
        # find index in image that corresponds to the center point of the circle
        x0 = obj["x0"]
        y0 = obj["y0"]
        r = obj["r"]

        # get index corresponding to middle of object
        # TODO floating point errors can cause this to be erronous in edge cases where x0/y0 is exactly half way between discretizations.
        x0_indx = np.argwhere((np.abs(self.x_cell - x0) < self.dx/2)).astype(np.int64)
        y0_indx = np.argwhere((np.abs(self.y_cell - y0) < self.dy/2)).astype(np.int64)

        # TODO: make faster by doing octants (instead of quarters)
        # Calculates circle in first quarter (quadrant I) then mirrors to other quadrants

        for i in range(math.ceil((r)/(self.dy))):
            for j in range(math.ceil((r)/(self.dx))):
                if (((self.x_cell[j + x0_indx] - x0)/r)**2 + ((self.y_cell[i + y0_indx] - y0)/r)**2 - 1) <= 0:
                    image[i + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, -j -1 + x0_indx] = value
                    image[i + y0_indx, -j -1 + x0_indx] = value


    def add_ellipse(self, obj, image, value):
        """Create image of a ellipse with specified value.

        Args:
            - obj (dict): model configuration for object (contains (x0,y0), r1, r2, er, sig)
            - image (np.ndarray): 2D array to be written to (note: overwrites contents when adding ellipse contents)
            - value (float): value to be written for ellipse
        """
        x0 = obj["x0"]
        y0 = obj["y0"]
        r1 = obj["r1"]
        r2 = obj["r2"]

        # get index corresponding to middle of object
        x0_indx = np.argwhere((np.abs(self.x_cell - x0) < self.dx/2)).astype(np.int64)
        y0_indx = np.argwhere((np.abs(self.y_cell - y0) < self.dy/2)).astype(np.int64)

        # does quadrant I then mirrors to other quadrants
        for i in range(math.ceil((r2)/(self.dy))):
            for j in range(math.ceil((r1)/(self.dx))):
                if (((self.x_cell[j + x0_indx] - x0)/r1)**2 + ((self.y_cell[i + y0_indx] - y0)/r2)**2 - 1) <= 0:
                    image[i + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, -j -1 + x0_indx] = value
                    image[i + y0_indx, -j -1 + x0_indx] = value

    def add_sinlike(self, obj, image, value):
        """Create image of a smooth sine-like distriubtion with specified peak.

        Args:
            - obj (dict): model configuration for object (contains (x0,y0), r, r, er, sig)
            - image (np.ndarray): 2D array to be written to (note: overwrites contents when adding ellipse contents)
            - value (float): value to be written for ellipse
        """
        x0 = obj["x0"]
        y0 = obj["y0"]
        r = obj["r"]

        x_matrix = np.tile(self.x_cell, (self.y_cell.size, 1))
        y_matrix = np.tile(self.y_cell, (self.x_cell.size, 1))

        # calculate the distance from center of distribution
        rho = np.sqrt((x_matrix.T - x0)**2 + (y_matrix - y0)**2)

        # remove distance that are oustide the diameter
        idx = (rho > r)
        rho[idx] = r
        # calculate sinlike
        sinelike = (value - image[0][0]) * np.cos(rho * 0.5 * math.pi / (r))
        #idx = (sinelike > 0.01)
        #sinelike[idx] = (value - image[0][0])

        image += sinelike
    
    def add_square(self, obj, image, value):
        """Function to add inhomogeneous square object to model. Object property varies linearly in x and y.

        Args:
            - obj (dict): model configuration for object (contains (x0,y0), r, r, er, sig)
            - image (np.ndarray): 2D array to be written to (note: overwrites contents when adding ellipse contents)
            - value (float): value to be written for ellipse
            
        """
        x0 = obj["x0"]
        y0 = obj["y0"]
        r = obj["r"] # half of square side length i.e. similar to radius

        # calculates the number of steps and corresponding er amount
        num_steps = int( np.ceil(r / self.dx))
        er_step = (value - image[0][0])/num_steps

        square = np.zeros((self.y_cell.size, self.x_cell.size))

        # creates squares of different sizes starting with the largest then overwriting the smaller ones
        for i in range(num_steps):
            indx = np.intersect1d(np.argwhere(self.x_cell > (x0 + i * self.dx - r)), np.argwhere(self.x_cell < (x0 - i * self.dx + r)))
            indy = np.intersect1d(np.argwhere(self.y_cell> (y0 + i * self.dy - r)), np.argwhere(self.y_cell < (y0 - i * self.dy + r)))
            square[indy[0]:indy[-1]+1, indx[0]:indx[-1]+1] = er_step*(i+1)

        image += square


    def write_image(self, image, prop):
        """ Writes image (2D np array) into image domain of model.
            Note for er_imag, it is assumed to be a real, positive number
        Args:
            - image (np.ndarray): 2D array to be written
            - prop (str): model property of image (i.e. 'er' or 'sig')
        """

        if not(image.shape[0] == self.image_domain.y_cell.size) or not(image.shape[1] == self.image_domain.x_cell.size):
            raise ValueError("Image must be the same size as the image domain")
        
        x_indx1 = np.argwhere(self.x_cell - self.image_domain.x_cell[0] >= -0.001)
        x_indx2 = np.argwhere(self.x_cell - self.image_domain.x_cell[-1] <= 0.001)
        x_indx = np.intersect1d(x_indx1, x_indx2)
        y_indx1 = np.argwhere(self.y_cell - self.image_domain.y_cell[0] >= -0.001)
        y_indx2 = np.argwhere(self.y_cell - self.image_domain.y_cell[-1] <= 0.001)
        y_indx = np.intersect1d(y_indx1, y_indx2)

        if prop == 'er':
            indx = (image < 1)
            image[indx] = 1.0
            self.er[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] = image
        elif prop == 'sig':
            indx = (image < 0)
            image[indx] = 0.0
            self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] = image
        elif prop == 'er_imag':
            # assumes er'' is positive real number
            indx = (image < 0)
            image[indx] = 0
            self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] = self.er_imag_to_sig(image)
        else:
            raise ValueError("prop value " + prop + " not supported")

    def add_image(self, image, prop):
        """ Adds image (2D np array) into image domain of model
        Note images can be non-physical (i.e. negative er)
        Args:
            - image (np.ndarray): 2D array to be written
            - prop (str): model property of image (i.e. 'er' or 'sig')
        """

        if not(image.shape[0] == self.image_domain.y_cell.size) or not(image.shape[1] == self.image_domain.x_cell.size):
            raise ValueError("Image must be the same size as the image domain")
        
        x_indx1 = np.argwhere(self.x_cell - self.image_domain.x_cell[0] >= -0.001)
        x_indx2 = np.argwhere(self.x_cell - self.image_domain.x_cell[-1] <= 0.001)
        x_indx = np.intersect1d(x_indx1, x_indx2)
        y_indx1 = np.argwhere(self.y_cell - self.image_domain.y_cell[0] >= -0.001)
        y_indx2 = np.argwhere(self.y_cell - self.image_domain.y_cell[-1] <= 0.001)
        y_indx = np.intersect1d(y_indx1, y_indx2)

        if prop == 'er':
            self.er[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] += image
        elif prop == 'sig':
            self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] += image
        elif prop == 'er_imag':
            self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] += self.er_imag_to_sig(image)
        else:
            raise ValueError("prop value " + prop + " not supported")
    
    def replace_background(self, old_er, old_sig, new_er, new_sig):
        """
        """
        self.er[self.er == old_er] = new_er
        self.sig[self.sig == old_sig] = new_sig


    def plot(self, image, title):
        """Plot model using image (er, sig)
        """
        plt.imshow(image, extent = [self.x[0], self.x[-1], self.y[0], self.y[-1]], origin = 'lower')
        self.rx.plot()
        ax = plt.gca()
        ax.set_xticks(self.x)
        ax.set_yticks(self.y)
        plt.title(title)
        plt.xticks(self.x)
        plt.yticks(self.y)
        plt.grid()
        plt.colorbar()
        plt.show()


    def plot_er(self):
        """Plot model object using permittivity
        """
        self.plot(self.er, 'Permittivity')
    
    def plot_sig(self):
        """Plot model object using conductivity
        """
        self.plot(self.sig, 'Conductivity')

    def translate(self,dx,dy):
        """ Translate model and objects by dx, dy
        Args:
            - dx (float): x translation
            - dy (float): y translation
        """

        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy
        self.rx.translate(dx,dy)
        self.image_domain.translate(dx,dy)

    def get_image(self, prop):
        """ Get prop ('er' or 'sig') image of model inside imaging domain

        Args:
            - prop (str): property to get ('er' or 'sig')
        Outputs:
            - image (np.ndarray): 2D array of model properties within imaging domian
        """

        x_indx1 = np.argwhere(self.x_cell - self.image_domain.x_cell[0] >= -0.001)
        x_indx2 = np.argwhere(self.x_cell - self.image_domain.x_cell[-1] <= 0.001)
        x_indx = np.intersect1d(x_indx1, x_indx2)
        y_indx1 = np.argwhere(self.y_cell - self.image_domain.y_cell[0] >= -0.001)
        y_indx2 = np.argwhere(self.y_cell - self.image_domain.y_cell[-1] <= 0.001)
        y_indx = np.intersect1d(y_indx1, y_indx2)

        if prop == 'er':
            image = self.er[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1]
        elif prop == 'sig':
            image = self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1]
        elif prop == 'er_imag':
            # returns real value image
            image = self.sig_to_er_imag(self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1])
        elif prop == "comp_er":
            # returns complex value
            image = self.er[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] - 1j * self.sig_to_er_imag(self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1])
        elif prop == "contrast":
            image = (self.er[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1] -1j * self.sig_to_er_imag(self.sig[y_indx[0]:y_indx[-1]+1, x_indx[0]:x_indx[-1]+1]) - self.er_b)/self.er_b
        else:
            raise ValueError("prop value " + prop + " not supported")
        
        return image

    def er_imag_to_sig(self, image):
        """Convert imaginary component of permittivity to conductivity.
        Args:
            - self (class): model class with frequency info
            - image (nd.ndarray): imaginary part of permittivity 
        """
        return image * 2 * np.pi* self.freq * constants.E0
    
    def sig_to_er_imag(self, image):
        """Convert conductivity to imaginary component of permittivity. Returns imaginary number with correct sign (for positive sigma)"""
        return image / (2*np.pi * self.freq * constants.E0)

    def compare_to_image(self, image, prop, do_plot):
        """ Compares model image to another image. Plots cross sections and gives root of sum square errors of cross sections.

        Args:
            - image (np.ndarray): 2D image 
            - prop (str): model property to use (e.g. 'er' or 'sig')
            - do_plot (bool): do plot?
        Outputs:
            - x_rsse and y_rsse
        """

        x_cross1= self.get_cross_section(self.get_image(prop), 0)
        x_cross2 = self.get_cross_section( image, 0)
        y_cross1= self.get_cross_section(self.get_image(prop), 1)
        y_cross2 = self.get_cross_section( image, 1)

        actual_image = self.get_image(prop)
        indx = np.argwhere(actual_image > self.er_b)
        if indx.size == 0:
            indx = np.argwhere(actual_image < self.er_b)
        profile = actual_image[indx[:,0],indx[:,1]]

        rsse1 = np.sqrt(np.sum(np.abs(x_cross1 - x_cross2)**2)/np.sum(x_cross1)**2)
        rsse2 = np.sqrt(np.sum(np.abs(y_cross1 - y_cross2)**2)/np.sum(y_cross1)**2)
        rsse_profile = np.sqrt(np.sum(np.abs(profile- image[indx[:,0],indx[:,1]])**2)/np.sum(profile**2))
        rsse_total = np.sqrt(np.sum(np.abs(actual_image- image)**2)/np.sum(actual_image**2))

        if do_plot:
            plt.plot(self.image_domain.x_cell, x_cross1, label = 'Image 1')
            plt.plot(self.image_domain.x_cell, x_cross2, label = 'Image 2')
            plt.xlabel('x position [m]')
            plt.xticks(self.image_domain.x)
            plt.ylabel(prop)
            plt.legend()
            plt.grid()
            plt.show()

            plt.plot(self.image_domain.y_cell, y_cross1, label = 'Image 1')
            plt.plot(self.image_domain.y_cell, y_cross2, label = 'Image 2')
            plt.xlabel('y position [m]')
            plt.xticks(self.image_domain.y)
            plt.ylabel(prop)
            plt.legend()
            plt.grid()
            plt.show()

        return (rsse1, rsse2, rsse_profile, rsse_total)

    @staticmethod
    def get_cross_section(image, axis):
        """ Get the cross section of two images along given axis

        Args:
            - image (np.ndarray): 2D array
            - axis (int): axis to hold steady (i.e. axis = 0, constant y, so plot across x)
        Outputs:
            - array (np.ndarray): 1D array of cross section of image 
        """

        if not(image.ndim == 2):
            raise ValueError('Image must be 2D')
            
        if not(axis < 2) or axis < 0:
            raise ValueError('Invalid axis for cross section')

        idx = int(image.shape[axis]/2)

        if axis == 0:
            array = image[idx,:]
        else:
            array = image[:,idx]

        return array
        
class ImageDomain():
    """Class for informationa and functions to do with the image domain"""

    def __init__(self, domain_config, scale):
        """Initialize ImageDomain using configuration data
        Args:
            - domain_config (dict): configuration data
            - scale (float): linear scale factor (use to create alternative grid)
        """
        self.dx = domain_config["dx"] * (1 + scale)
        self.dy = domain_config["dy"] * (1 + scale)
        self.x1 = domain_config["x1"] * (1 + scale/2)
        self.x2 = domain_config["x2"] * (1 + scale/2)
        self.y1 = domain_config["y1"] * (1 + scale/2)
        self.y2 = domain_config["y2"] * (1 + scale/2)
        self.freq = np.asarray(domain_config["recon_freq"])
        self.ex_angle = domain_config["exclu_angle"] * np.pi / 180

    @property
    def x_size(self):
        return self.x2 - self.x1

    @property
    def y_size(self):
        return self.y2 - self.y1

    @property
    def x(self):
        return np.linspace(self.x1, self.x2, num = int(round(self.x_size/self.dx, 1)) +1)
    @property
    def y(self):
        return np.linspace(self.y1, self.y2, num = int(round(self.y_size/self.dy, 1)) +1)
    @property
    def x_cell(self):
        return np.linspace(self.x1 + self.dx/2, self.x2 - self.dx/2, num = int(round(self.x_size/self.dx, 1)))
    @property
    def y_cell(self):
        return np.linspace(self.y1 + self.dy/2, self.y2 - self.dy/2, num = int(round(self.y_size/self.dy, 1)))
    @property
    def area(self):
        return self.y_size*self.x_size

    def translate(self, dx, dy):
        """ Translate model and objects by dx, dy
        Args:
            - dx (float): x translation
            - dy (float): y translation
        """
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy



