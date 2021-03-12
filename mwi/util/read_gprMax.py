import os
import numpy as np
import h5py
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import czt

def read_out_data(folder):
    """Function to read rx data from gprMax (.out files). Currently only reads in Ez data.
    Args:
        - model (class): class holding information about simulation
        - folder (string): file path to folder holding .out files

    Outupts:
        - rx_data (np.ndarray): ntx x nrx x N array with rx_data
        - time (np.ndarray): time series
        - rx_data_f (np.ndarray): ntx x nrx x N/2 array with complex rx_data in frequency domain
        - freq (np.ndarray): frequency series
    """
    files = os.listdir(folder)
    out_files = [f for f in files if ".out" in f]

    filepath = os.path.join(folder, out_files[0])
    f = h5py.File(filepath, 'r')
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']
    nrx = f.attrs['nrx']
    f.close()

    # 50 MHz steps
    freq = np.arange(1e8, 10e9 + 5e7, step = 5e7)
    nf = freq.size

    time = np.linspace(0, (iterations - 1) * dt, num=iterations)
    rx_data = np.zeros((nrx, len(out_files),  iterations))
    rx_data_f = np.zeros((nrx, len(out_files),  nf), dtype = np.complex_)

    i = 0
    for file in out_files:
        for j in range(nrx):
            filepath = os.path.join(folder, file)
            f = h5py.File(filepath, 'r')
            # hdf5 data path
            path = "/rxs/rx" + str(j+1) + "/"
            # read in Ez data
            rx_data[j, i, :] = f[path]["Ez"]
            # do fft and remove negative frequencies
            #rx_data_f[j, i, :] = np.fft.fftshift(np.fft.fft(rx_data[j, i, :]))[int(iterations/2):-1]
            _,freq_data = czt.time2freq(time, rx_data[j,i,:], freq)
            rx_data_f[j,i,:] = freq_data

        i += 1
    
    #freq = np.fft.fftshift(np.fft.fftfreq(iterations, dt))[int(iterations/2):-1]
    return (rx_data, time, rx_data_f, freq)

def read_snapshots(folder):
    """Function to read field data (snapshots) from gprMax (.vti files). Currently only reads in Ez data.
    Args:
        - model (class): class holding information about simulation
        - folder (str): 

    Outputs:
        - field_data (np.ndarray): ntx x nrx x ny x nx array with field data
    """
    # this value is hard coded in the gprMax setup.... TODO define dynamically (maybe read from .in file?)
    dt = 5e-11

    # get the snapshot folders, one for each tx
    files = os.listdir(folder)
    snap_folders = [f for f in  files if "_snaps" in f]

    ntx = len(snap_folders)

    if ntx == 0:
        raise FileNotFoundError('Could not find snapshot folder')

    # get snapshot files (.vti) in the first folder. one snapshot for each iteration saved
    snapshots = os.listdir(os.path.join(folder, snap_folders[0]))
    snapshots = [f for f in snapshots if ".vti" in f]
    num = len(snapshots)

    if num == 0:
        raise FileNotFoundError('Could not find .vti files')

    # read in one .vti to get image size
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(os.path.join(folder, snap_folders[0], snapshots[0]))
    reader.Update()
    image = reader.GetOutput()

    nx = image.GetDimensions()[0] -1
    ny = image.GetDimensions()[1] -1

    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid image dimensions. Must be greater than 0")
    
    # 50 MHz steps
    freq = np.arange(1e8, 10e9 + 5e7, step = 5e7)
    nf = freq.size 

    # initialize arrays
    time = np.linspace(0, (num - 1) * dt, num=num)
    data = np.zeros((ntx, num, ny, nx))
    data_f = np.zeros((ntx, nf, ny, nx), dtype = np.complex_)

    # iterate through each folder and each .vti file
    i = 0
    for snap_folder in snap_folders:
            for j in range(len(os.listdir(os.path.join(folder, snap_folder)))):
                # setup vtk read
                reader = vtk.vtkXMLImageDataReader()
                reader.SetFileName(os.path.join(folder, snap_folder, "snapshot" + str(j+1) + ".vti"))
                reader.Update()
                # convert vti image to numpy array (Ez is component 3)
                image = reader.GetOutput().GetCellData().GetArray("E-field")
                array = vtk_to_numpy(image)
                array = array[:,2]
                data[i, j, :, :] = array.reshape((ny, nx))
            i += 1

    # do fft of each pixel over time. TODO probably a faster way to do this...
    for i in range(ntx):
        for j in range(ny):
            for k in range(nx):
                #data_f[i, :, j, k] = np.fft.fftshift(np.fft.fft(data[i, :, j, k]))[int(num/2):-1]
                _,data_f[i, :, j, k] = czt.time2freq(time, data[i,:, j,k], freq)
    
    
    #freq = np.fft.fftshift(np.fft.fftfreq(num, dt))[int(num/2):-1]

    return (data, time, data_f, freq)