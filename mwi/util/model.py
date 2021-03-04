import numpy as np
import math
from matplotlib import pyplot as plt

class Model():
    """Class for simulation model: permittivity and conductivity images, measurement surface, and imaging domain

    
    Known issues:
        -placing object at mid point between discretizations can cause errors due to float precision.
    """

    def __init__(self, config, rx_surface, image_domain):
        """Initialize class using configuration data, measurement surface, and imaging domain vector.

        Args:
            - config (dict): model configuration settings (see example\model_config.json)
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

        # create 2D image using configuration data
        self.er = self.create_image(config, 'er')
        self.sig = self.create_image(config, 'sig')

        # assign measurement surface
        self.rx = rx_surface

        # assign image domain
        self.image_domain = image_domain

    #properties
    @property
    def x_size(self):
        return self.x2 - self.x1
    @property
    def y_size(self):
        return self.y2 - self.y1
    @property
    def x(self):
        return np.linspace(self.x1, self.x2, num = math.ceil(self.x_size/self.dx) +1)
    @property
    def y(self):
        return np.linspace(self.y1, self.y2, num = math.ceil(self.y_size/self.dy) +1)
    @property
    def x_cell(self):
        return np.linspace(self.x1 + self.dx/2, self.x2 - self.dx/2, num = math.ceil(self.x_size/self.dx))
    @property
    def y_cell(self):
        return np.linspace(self.y1 + self.dy/2, self.y2 - self.dy/2, num = math.ceil(self.y_size/self.dy))
    @property
    def nrx(self):
        return self.rx.nrx
    @property
    def ntx(self):
        return self.rx.ntx

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
                pass
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
        # TODO floating point errors can cause this to erronous in edge cases where x0/y0 is exactly half way between discretizations.
        x0_indx = np.argwhere((np.abs(self.x_cell - x0) < self.dx/2)).astype(np.int64)
        y0_indx = np.argwhere((np.abs(self.y_cell - y0) < self.dy/2)).astype(np.int64)

        # TODO: make faster by doing octants (instead of quarters)
        # Calculates circle in first quarter (quadrant I) then mirrors to other quadrants
        for i in range(math.ceil((y0 + r)/(self.dy))):
            for j in range(math.ceil((x0 + r)/(self.dx))):
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
        for i in range(math.ceil((y0 + r2)/(self.dy))):
            for j in range(math.ceil((x0 + r1)/(self.dx))):
                if (((self.x_cell[j + x0_indx] - x0)/r1)**2 + ((self.y_cell[i + y0_indx] - y0)/r2)**2 - 1) <= 0:
                    image[i + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, j + x0_indx] = value
                    image[-i -1 + y0_indx, -j -1 + x0_indx] = value
                    image[i + y0_indx, -j -1 + x0_indx] = value

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

class ImageDomain():
    """Class for informationa and functions to do with the image domain"""

    def __init__(self, domain_config):
        """Initialize ImageDomain using configuration data
        Args:
            - domain_config (dict): configuration data
        """
        self.dx = domain_config["dx"]
        self.dy = domain_config["dy"]
        self.x1 = domain_config["x1"]
        self.x2 = domain_config["x2"]
        self.y1 = domain_config["y1"]
        self.y2 = domain_config["y2"]

    @property
    def x_size(self):
        return self.x2 - self.x1

    @property
    def y_size(self):
        return self.y2 - self.y1

    @property
    def x(self):
        return np.linspace(self.x1, self.x2, num = math.ceil(self.x_size/self.dx) +1)
    @property
    def y(self):
        return np.linspace(self.y1, self.y2, num = math.ceil(self.y_size/self.dy) +1)
    @property
    def x_cell(self):
        return np.linspace(self.x1 + self.dx/2, self.x2 - self.dx/2, num = math.ceil(self.x_size/self.dx))
    @property
    def y_cell(self):
        return np.linspace(self.y1 + self.dy/2, self.y2 - self.dy/2, num = math.ceil(self.y_size/self.dy))

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


