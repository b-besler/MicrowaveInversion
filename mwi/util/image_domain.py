
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

    

