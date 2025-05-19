"""
Example usage that illustrates the basic usage of phonopyReaders.py
and that generates some simple figures

Written by Matthew Houtput (matthew.houtput@uantwerpen.be)
"""

### PREAMBLE

## Import necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from phonopyReaders import PhonopyCommensurateCalculation, YCalculation
from pathsLabels import get_path_and_labels

## Set plot options for matplotlib
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble']="\\usepackage{bm}"
mpl.rcParams['hatch.linewidth']=4.0


### PHONON FREQUENCIES

## Load the supercell calculation
yaml_filename = "data/LiF/LiF_a4.004_E0.0z_super444.yaml"
born_filename = "data/LiF/BORN_LiF_a4.004.txt"
calc = PhonopyCommensurateCalculation(yaml_filename, born_filename)

## Plot the phonon bands
path, path_labels = get_path_and_labels("FCC", break_z=False)
calc.plot_bands(path, path_labels, include_nac="Gonze", unit="THz",
                save_filename="test/phonon_bands")


### NONLINEAR INTERACTION

## Load the supercell calculations with electric field
yaml_filenames = ["data/LiF/LiF_a4.004_E-0.01z_super444.yaml",
                  "data/LiF/LiF_a4.004_E0.0z_super444.yaml",
                  "data/LiF/LiF_a4.004_E0.01z_super444.yaml"]
Efields = [-0.01, 0.0, 0.01]  # units of V/Å
born_filename = "data/LiF/BORN_LiF_a4.004.txt"
calc2 = YCalculation(yaml_filenames, Efields, born_filename, take_imag=True)

## Make a plot of |Y_{nu,nu',z}|^2
path2, path_labels2 = get_path_and_labels("FCC", break_z=True)
calc2.plotY(path2, path_labels2, include_nac="Gonze", subplots=(2,3),
            save_filename = "test/Yplot")

## Calculate the 1-electron-2-phonon spectral function
q_mesh_size = 16  # Larger values allow smaller sigmas, but take much longer
sigma = 0.8  # Set to smaller values for more accurate calculation
temperature = 0  # units of K
results = calc2.calculate_Tomega(q_mesh_size = q_mesh_size, sigma = sigma, 
                                 include_nac="Gonze", temperature = temperature,
                                 q_split_levels = 1, parallel_jobs = 4)
results.save_npz("test/Tomega_data")
results.plot(save_filename="test/Tomega_plot")


## Display all the figures
plt.show()
