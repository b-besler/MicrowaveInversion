import numpy as np
import os
import pathlib
import h5py
import subprocess
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
            make_geometry(model, folder)
            geom = True

    # translate model so extent is >=0 (gprMax uses >=0 extent)
    model.translate(-model.x[0], -model.y[0])
        
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

def make_geometry(model, folder):
        """Create geometry files for gprMax. Consists of text file with material definition and hdf5 images with material indices.

        Arg:
            model: model object with er/sig images to be written
        """
        textFileName = os.path.join(folder,model.name,model.name +"_geometry.txt")
        hdf5FileName = os.path.join(folder,model.name,model.name +"_geometry.h5")
        
        # round to three decimal places to avoid too many unique permittivities
        er = np.round(model.er, 3)
        sig = np.round(model.sig, 3)

        unique_er = np.unique(er) #find unique values
        E = unique_er.size #number of unique values
        unique_sig = np.unique(sig)
        S = unique_sig.size

        data = np.zeros((er.shape[0],er.shape[1])) #initialize arrays
        mat = np.zeros((E*S,er.shape[0],er.shape[1]))

        if not os.path.isdir(os.path.join(folder, model.name)):
            os.mkdir(os.path.join(folder, model.name))

        f = open(textFileName,"w")
        
        k = 0 #material index
        for i in range(E):
            for j in range(S):
                indx_e = (er == unique_er[i]) #image of true where er == unique_er
                indx_s = (sig == unique_sig[j])
                indx = np.logical_and(indx_e,indx_s) # image of true where er == unique_er and sig == unique_sig
                # Only write materials with corresponding er and sig
                if(np.any(indx)):
                    # Check whether materials are free space or not
                    if (abs(unique_er[i]-1) > 0.00001) or (abs(unique_sig[j]-0) > 0.00001): #if properties aren't that of free space
                        f.write("#material: %f %f 1 0 mat%d\n" %(unique_er[i], unique_sig[j], k)) #write material properties and name (mat[k]) to text
                        mat[(i+1)*(j+1)-1,::] = indx.astype(int)*k  #write material index (k) to image, one image per material (0 where no index)
                        k = k+1
                    else:
                        mat[(i+1)*j,::] = indx.astype(int)*-1 #free space is index -1

        f.close()
        data = np.sum(mat,0) #combine material images into one image
        data = data.T
        write_hdf5(model, hdf5FileName, data)  #write material indices to hdf5

def write_hdf5(model, name,data):
    """Make hdf5 image with gprMax material indices (1-n for each unique material) to define gprMax geometry
        
    Args:
        -model (class): model with information for file format
        -name (string): full file path including file name and extension for hdf5 file
        -data (np.ndarray): 2D array with material indices

    Returns:
        Nothing, but write data to hdf5 with proper formatting for gprMax
    """
    
    data2 = np.zeros((data.shape[0],data.shape[1],1),dtype = np.int16)
    data2[:,:,0]= data.astype(np.int16)
    E = np.zeros((12,model.er.shape[0],model.er.shape[1],1)) #dummy data
    dl = np.array([model.dx,model.dy,model.dx])

    # hdf5 format is specific to gprMax
    # data holds hdf5 image of material indicies, rigidE/H are dummy data, attributes are required metadata
    f = h5py.File(name,'w')
    f.create_dataset("data",data=data2)
    f.create_dataset("rigidE",data=E)
    f.create_dataset("rigidH",data=E)
    f.attrs['gprMax']="3.1.5"
    f.attrs['dx_dy_dz']=dl
    f.attrs['title'] = model.name
    f.close()

def run(model, folder):
    """Runs all simulations in folder using model data
    Args:
        - model (class): model class holding information corresponding to simulation model
        - folder (str): folder holding all the simulation files
    """
    
    k = 0
    for i in range(model.ntx):
        if (os.path.exists(os.path.join(folder, model.name, model.name + '_output', model.name + "_Tx" + str(i) + ".out"))):
            k += 1
        else:
            os.chdir("gprMax")
            result = subprocess.run("python -m gprMax " + os.path.join(folder, model.name, model.name + "_Tx" + str(i) +".in"))
            os.chdir("..")
            if not (result.returncode == 0):
                raise ValueError("gprMax simulation run failed.")
    if not (k==0):
        print("Skipped "+ str(k) + " simulations in " + model.name + " because they have results.")    