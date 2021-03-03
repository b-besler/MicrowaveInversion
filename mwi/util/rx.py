class MeasurementSurface():
    """Class for circular measurement surface defined by center (x0,y0), radius (r), and number of receivers/transmitters (rx/tx)
    """

    def __init__(self, config):
        """Initialize measurement surface class with configuration information.
        
        Args:
            - config (dict): measurement configuration file (.json)
        """

        self.x0 = config["x0"]
        self.y0 = config["y0"]
        self.nrx = config["nr"]
        self.ntx = config["nt"]
        self.r = config["r"]
    
    
        