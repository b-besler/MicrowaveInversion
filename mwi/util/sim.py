import numpy as np
import os
import pathlib

# functions for gprMax simulation


class Source():
    """Class for gpr max simulation source"""

    def __init__(self, source_config):
        """Initialize source using config settings
        Args:
            - source_config (dict): configuration file with source settings        
        """

        self.type = source_config["type"]
        self.t0 = source_config["t0"]
        self.f0 = source_config["f0"]
        self.amp = source_config["amp"]

def make(model, src, folder):
    """Create gprMax simulation files for model object. Consists of *in text files with gprMax commands and geometry files
    Args:
        - model (class): class with rx/tx locations, er/sig images, and image domain information
        - src (class): class holding source information
        - folder (string): output folder to make simulation files in
    """
    
    folderPath = os.path.join(folder,model.name)
    # create new folder with model name
    if not os.path.exists(folderPath):
        pathlib.Path(folderPath).mkdir(parents=True,exist_ok=True)
        
    geom = False
    # if not just free space we have to make a geometry file
    if (np.any(model.er-1 > 0.00001) and np.any(model.sig-0 < 0.00001)):
            #sim.make_geometry(model)
            geom = True
        
    #calculate rx/tx locations
    tx = model.rx.calc_tx_discrete(model.dx,model.dy)
    rx = model.rx.calc_rx_discrete(model.dx,model.dy)

    #different simulation file (*.in) for each tx
    for i in range(model.ntx): 
        # write gprMax simulation file (see example/gpr_max_sim) and gprMax docs for details.
        fileName = os.path.join(folderPath,model.name + '_Tx' + str(i)+".in")
            
        f = open(fileName,"w")
            
        f.write("#title: %s\n" % model.name) #write title
        f.write("#domain: %f %f %f\n" % (model.x_size, model.y_size, model.dx)) #domain extent command
        f.write("#dx_dy_dz: %f %f %f\n" % (model.dx, model.dx, model.dx)) #discretization command
        f.write("#time_window: %e\n" % src.t0) #time window command, length t0
        f.write("#waveform: %s %f %e my_gauss\n" %(src.type, src.amp, src.f0)) #waveform of type, amplitude, and freq
        
        if geom: #if we have a geometry file, include command 
            f.write("#geometry_objects_read: 0 0 0 %s.h5 %s.txt\n" % (model.name+"_geometry",model.name+"_geometry"))

        f.write("#hertzian_dipole: z %f %f 0 my_gauss\n" %(tx[0][i], tx[1][i])) #create tx at correct location 
        #place rx at correct locations
        for j in range(model.nrx):
            f.write("#rx: %f %f %f\n" %(rx[0][j], rx[1][j],0))
            
        #snapshots - uses python inside of the gpr command, look at example *.in file and gprMax docs
        f.write("#python:\nfrom gprMax.input_cmd_funcs import * \n")
        f.write("import numpy as np\ndt = 5e-11 \nI = np.floor(float(%e*0.1/dt))\n" % src.t0)
        f.write("N = np.floor(float(%e/dt))\nfor i in range(1,int(N)):\n"% src.t0)
        f.write("\tsnapshot(%f,%f,0,%f,%f,%f,%f,%f,%f,i*dt,\'snapshot\'+str(i))\n#end_python:\n"%(
            (model.x_size - model.image_domain.x_size)/2,
            (model.y_size - model.image_domain.y_size)/2,
            (model.x_size + model.image_domain.x_size)/2,
            (model.y_size + model.image_domain.y_size)/2,
            model.dx,
            model.dx,
            model.dx,
            model.dx
        ))

        # extent of geometry view
        f.write("#geometry_view: 0 0 0 %f %f %f %f %f %f %s n\n"%(model.x_size,model.y_size,model.dx,model.dy,model.dx,model.dx,model.name + '_Tx' + str(i)))
        # output directory (only *.out receiver data)
        f.write("#output_dir: %s" % (model.name + "_output"))