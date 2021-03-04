import numpy as np


class Source():
    """Source class gpr max sim
    """

    def __init__(self, source_config):
        """Initialize source using config settings
        Args:
            - source_config (dict): configuration file with source settings        
        """

        self.type = source_config["type"]
        self.t0 = source_config["t0"]
        self.f0 = source_config["f0"]
        self.amp = source_config["amp"]

        