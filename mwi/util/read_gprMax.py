import os
import numpy as np
import h5py
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

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
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    rx_data = np.zeros((len(out_files), nrx, iterations))
    rx_data_f = np.zeros((len(out_files), nrx, int(iterations/2)), dtype = np.complex_)
    print(folder)
    
    i = 0
    for file in out_files:
            filepath = os.path.join(folder, file)
            f = h5py.File(filepath, 'r')
            # hdf5 data path
            path = "/rxs/rx" + str(i+1) + "/"
            # read in Ez data
            rx_data[i, :, :] = f[path]["Ez"]
            print(np.ceil(iterations/2))
            rx_data_f[i, :, :] = np.fft.fftshift(np.fft.fft(rx_data[i, :, :]))[:,int(iterations/2):-1]
            i += 1
    
    freq = np.fft.fftshift(np.fft.fftfreq(iterations, dt))[int(iterations/2):-1]

    return (rx_data, time, rx_data_f, freq)