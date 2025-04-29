"""
Example usage that generates all figures in arXiv:2412.09470

Written by Matthew Houtput (matthew.houtput@uantwerpen.be)
"""

#%% Load important libraries

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from phonopyReaders import PhonopyCommensurateCalculation, YCalculation, \
    TomegaResults, round_plot_range
from pathsLabels import get_path_and_labels

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble']="\\usepackage{bm}"
mpl.rcParams['hatch.linewidth']=4.0
np.set_printoptions(precision=6, suppress=True)

#%% Input parameters
plots_folder = "plots/"
text_sizes=(15, 16, 18)
recalculate = False  # Warning: recalculating T(omega) may take several days
recalc_qmesh = 128  # Set to 16 for fast inaccurate calculation
recalc_sigma = 0.1  # Smearing width, set to 0.8 when setting recalc_qmesh=16
recalc_parallel_jobs = 4

def read_filename(name):
    dirname, tail = os.path.split(name)
    filename, extension = os.path.splitext(tail)
    components = filename.split("_")
    material_props = {
        "material_name": components[0],
        "aval": float(components[1][1:]),
        "Efield": float(components[2][1:]),
        "supercell_size": [int(x) for x in components[3][5:]],
        "directory": dirname,
        "file_type": extension
    }
    return material_props


#%% Plot of the FCC Brillouin zone with high-symmetry path

data_folder = "data/LiF"
calc = PhonopyCommensurateCalculation(
    data_folder+"/LiF_a4.004_E0.0z_super666.yaml", 
    born_filename=data_folder+"/BORN_LiF_a4.004.txt")
path, path_labels = get_path_and_labels("FCC", False)
# Shift the labels slightly so they're easily readable on the final figure
label_shifts =  [
                    [-0.05, 0.0, -0.05],    # Gamma
                    [0.02, -0.08, -0.1],    # X
                    [-0.08, 0.0, 0.02],     # W
                    [-0.08, 0.0, 0.02],     # K
                    [],
                    [0.15, 0.15, 0.15],     # L
                    [-0.01, 0.0, -0.11],    # U
                    [], [], [], [], [], []
                ]
label_style = dict(color="black", fontsize=14, horizontalalignment='center', 
                   verticalalignment='center_baseline')
reciprocal_lattice_vectors = None
quiver_labels = ["$\\bm{b}_1$", "$\\bm{b}_2$", "$\\bm{b}_3$"]
quiver_plot = np.empty((3, 3, 3))
# Data for the reciprocal lattice vectors:
quiver_plot[0:3,:,0] = 0.5*np.eye(3)
quiver_plot[0:3,:,1] = 0.2*np.eye(3)
quiver_plot[0:3,:,2] = 0.8*np.eye(3)

save_filename = plots_folder+"FCC1BZ_high"
view_angles = (18, 26, 0)
save_bbox = [[1.8, 0.9], [4.7, 4.5]]
calc.plot_Brillouin(path=path, path_labels=path_labels, view_angles=view_angles,
                    label_shifts=label_shifts, quiver_labels=quiver_labels, 
                    quiver_plot=quiver_plot, save_filename=save_filename, 
                    reciprocal_lattice_vectors=reciprocal_lattice_vectors, 
                    label_style=label_style, save_bbox_extents=save_bbox)

#%% Plot of the FCC Brillouin zone with low-symmetry path

data_folder = "data/LiF"
calc = PhonopyCommensurateCalculation(
    data_folder+"/LiF_a4.004_E0.0z_super666.yaml", 
    born_filename=data_folder+"/BORN_LiF_a4.004.txt")
path, path_labels = get_path_and_labels("FCC", True)
# Shift the labels slightly so they're easily readable on the final figure
label_shifts =  [
                    [-0.05, 0.0, -0.05],    # Gamma
                    [0.02, -0.08, -0.1],    # X
                    [-0.08, 0.0, 0.02],     # W
                    [-0.08, 0.0, 0.02],     # K
                    [],
                    [-0.09, -0.01, -0.08],   # X1
                    [0.0, -0.1, 0.0],       # U1
                    [0.15, 0.15, 0.15],     # L
                    [0.1, 0.0, 0.0],        # W1
                    [0.1, 0.1, 0.0],        # K1
                    [0.1, 0.0, 0.0],        # W2
                    [], [], []
                ]
label_style = dict(color="black", fontsize=14, horizontalalignment='center', 
                   verticalalignment='center_baseline')
reciprocal_lattice_vectors = None
quiver_labels = ["$\\bm{b}_1$", "$\\bm{b}_2$", "$\\bm{b}_3$", 
                 "$\\bm{\\mathcal{E}}$"]
quiver_plot = np.empty((4, 3, 3))
# Data for the reciprocal lattice vectors:
quiver_plot[0:3,:,0] = 0.5*np.eye(3)
quiver_plot[0:3,:,1] = 0.2*np.eye(3)
quiver_plot[0:3,:,2] = 0.8*np.eye(3)
# Data for the electric field:
quiver_plot[3, :, 0] = 0.5*np.array([1.0,1.0,0.0])
quiver_plot[3, :, 1] = 0.2*np.array([1.0,1.0,0.0])
quiver_plot[3, :, 2] = 0.75*np.array([1.0,1.0,0.0])

save_filename =  plots_folder+"FCC1BZ_low"
view_angles = (18, 26, 0)
save_bbox = [[1.8, 0.9], [4.7, 4.5]]
calc.plot_Brillouin(path=path, path_labels=path_labels, view_angles=view_angles,
                    label_shifts=label_shifts, quiver_labels=quiver_labels, 
                    quiver_plot=quiver_plot, save_filename=save_filename, 
                    reciprocal_lattice_vectors=reciprocal_lattice_vectors, 
                    label_style=label_style, save_bbox_extents=save_bbox)

#%% Plot of T(omega) convergence in LiF

unit="THz"
supercell_sizes = [2,3,4,5,6]
if recalculate:
    material_name = "LiF"
    moments = np.array([-0.5, -1.0, -1.5])
    a_str = "_a4.004"
    Efield = 0.01
    data_folder = "data/LiF"
    born_file = data_folder+"/BORN_"+material_name+a_str+".txt"
    common_str = data_folder+"/"+material_name+a_str+"_"
    print("Started LiF T(omega) calculations at T=0K")
    print("---------------------------------------------")
    for index, supercell_size in enumerate(supercell_sizes):
        supercell_string = "super"+3*str(supercell_size)
        Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
        Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
        Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
        Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                            np.array([-Efield,0.00,Efield]), 
                            born_filename=born_file, take_imag=True)
        savedata_filename = "results/LiF/"+supercell_string+"/qmesh"\
            +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str+"_T0"
        results = Ycalc.calculate_Tomega(
            q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit, 
            sigma=recalc_sigma, include_nac="Gonze", moments=moments, 
            q_split_levels=2, parallel_jobs=recalc_parallel_jobs, 
            savedata_filename = savedata_filename, temperature=0)
        print("Finished "+str(index+1)+" of "+str(len(supercell_sizes))\
              +" calculations")
    print("\n")
omega_max = 0
Tomega_max = 0
fig, ax = plt.subplots()
plot_handles = []
inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a4.004"
for index, supercell_size in enumerate(supercell_sizes):
    supercell_string = "super"+str(supercell_size)*3  # e.g. super666
    supercell_string2 = "x".join(str(supercell_size)*3)  # e.g. 6x6x6
    results = TomegaResults("results/LiF/"+supercell_string+"/"+\
                            inputs_string+"_T0.npz")
    data = results.to_dict()
    omega = data["omega"]
    T_omega = data["Tomega"]
    if index == 0:
        full_data_array = np.array([omega])
        header = "  Frequency (THz)      "
    full_data_array = np.append(full_data_array, np.array([T_omega]), axis=0)
    header += "    T(omega), "+(supercell_string2).ljust(11)

    omega_max = max(omega[-1], omega_max)   
    Tomega_max = max(np.max(T_omega), Tomega_max) 
    mesh_str = "$"+str(supercell_size)+"\\times"\
        +str(supercell_size)+"\\times"+str(supercell_size)+"$"
    plot_handle, = ax.plot(omega, T_omega, label = mesh_str)
    plot_handles.append(plot_handle)
    print("Moments of T(omega) for "+supercell_string+":")
    print(data['Tmoments'])

_, Tomega_max_scale = round_plot_range(0, Tomega_max, clamp_min = 0)
    
ax.set_title("LiF", size=text_sizes[2])
ax.set_xlabel("Frequency ("+unit+")", fontsize=text_sizes[1])
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
ax.set_xlim(0, omega_max)
ax.set_ylim(0, Tomega_max_scale)
ax.legend(handles=plot_handles, fontsize=text_sizes[0], loc="upper right")
ax.tick_params(axis='both', labelsize=text_sizes[0])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
fig.tight_layout()
fig.show()

fig.savefig(plots_folder+"LiF_Tomega_conv_"+inputs_string+".pdf")

np.savetxt(plots_folder+"/LiF_Tomega_conv_"+inputs_string+"_data.txt", 
           np.transpose(full_data_array), header=header)

#%% Plot of phonon bands and LATO weights in LiF

data_folder = "data/LiF"
calc = PhonopyCommensurateCalculation(
    data_folder+"/LiF_a4.004_E0.0z_super666.yaml", 
    born_filename=data_folder+"/BORN_LiF_a4.004.txt")
path, path_labels = get_path_and_labels("FCC", False)
path_low, path_labels_low = get_path_and_labels("FCC", True)
calc.plot_bands(path, path_labels, npoints=101, include_nac="Gonze", 
                plot_range=(0,20), title="LiF phonon bands", 
                save_filename=plots_folder+"LiF_phonons_a4.004_super666", 
                text_sizes=text_sizes)
calc.plot_LATO_weights(path_low, path_labels_low, npoints=101, num_markers=151,
                       include_nac="Gonze", plot_range=(0,20), 
                       save_filename=plots_folder+"LiF_phonons_a4.004_super666",
                       subplots=False, text_sizes=text_sizes)

#%% Plot of strength of |Y_{nu1,nu2}(q)|^2 in LiF

path, path_labels = get_path_and_labels("FCC", True)
data_folder = "data/LiF"
Efield = 0.01
born_file = data_folder+"/BORN_LiF_a4.004.txt"
Eminus_file = data_folder+"/LiF_a4.004_E-"+str(Efield)+"z_super666.yaml"
Ezero_file = data_folder+"/LiF_a4.004_E0.0z_super666.yaml"
Eplus_file = data_folder+"/LiF_a4.004_E"+str(Efield)+"z_super666.yaml"
savefig_filename = save_filename=plots_folder+"Yplots_LiF_super666_a4.004"
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for LiF"
Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                      np.array([-Efield,0.00,Efield]), born_filename=born_file,
                      take_imag=True)
plot_data = Ycalc.plotY(path, path_labels, include_nac="Gonze", npoints=101,
                        num_markers=151,
                        degenerate_cutoff=1e-3, subplots=None,
                        save_filename=savefig_filename,
                        plot_range=(0,20), title2=title_string,
                        text_sizes=text_sizes)
Y2_max = plot_data[0]
Y2_sum_max = plot_data[1]
print("Maximum value of |Y|² on the individual plots: "+str(Y2_max)+"Å²")
print("Maximum value of |Y|² on the summed plot: "+str(Y2_sum_max)+"Å²")

#%% Plot of LATO contributions to T(omega) in LiF

inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a4.004"
results = TomegaResults("results/LiF/super666/"+inputs_string+"_T0.npz")
data = results.to_dict()
results.save_txt(plots_folder+"LiF_Tomega_LATO_"+inputs_string+"_data",
                 plots_folder+"LiF_Tmoments_LATO_"+inputs_string+"_data")
fig, axes, _ = results.plot(title="LiF", text_sizes=text_sizes)
# Set scientific notation on the main axis:
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
# Rename the text on the inset:
axes[1].set_ylabel("Relative energy contribution", fontsize=text_sizes[0])
fig.tight_layout()
fig.show()
fig.savefig(plots_folder+"LiF_Tomega_LATO_"+inputs_string+".pdf")

#%% Plot of T(omega) for different temperatures in LiF

unit="THz"
temperatures = [0,50,100,150,200,250,300]
inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a4.004"
if recalculate:
    material_name = "LiF"
    moments = np.array([-0.5, -1.0, -1.5])
    a_str = "_a4.004"
    Efield = 0.01
    data_folder = "data/LiF"
    born_file = data_folder+"/BORN_"+material_name+a_str+".txt"
    common_str = data_folder+"/"+material_name+a_str+"_"
    Eminus_file = common_str+"E-"+str(Efield)+"z_super666.yaml"
    Ezero_file = common_str+"E0.0z_super666.yaml"
    Eplus_file = common_str+"E"+str(Efield)+"z_super666.yaml"
    Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                        np.array([-Efield,0.00,Efield]), 
                        born_filename=born_file, take_imag=True)
    print("Started LiF T(omega) temperature calculations for 6x6x6 supercell")
    print("-----------------------------------------------------------------")
    for index, temperature in enumerate(temperatures):
        savedata_filename = "results/LiF/super666/qmesh"\
            +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str\
            +"_T"+str(temperature)
        results = Ycalc.calculate_Tomega(
            q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit, 
            sigma=recalc_sigma, include_nac="Gonze", moments=moments, 
            q_split_levels=2, parallel_jobs=recalc_parallel_jobs, 
            savedata_filename = savedata_filename, temperature=temperature)
        print("Finished "+str(index+1)+" of "+str(len(temperatures))\
              +" calculations")
    print("\n")
omega_max = 0
Tomega_max = 0
fig, ax = plt.subplots()
plot_handles = []
for index, temperature in enumerate(temperatures):
    temp_str = "T"+str(temperature)
    data = np.load("results/LiF/super666/"+inputs_string+"_"+temp_str+".npz")
    omega = data["omega"]
    T_omega = data["Tomega"]  
    omega_max = max(omega[-1], omega_max)   
    Tomega_max = max(np.max(T_omega), Tomega_max) 
    legend_str = "$T = "+str(temperature)+"$K"
    plot_handle, = ax.plot(omega, T_omega, label = legend_str)
    plot_handles.append(plot_handle)
    if index == 0:
        full_data_array = np.array([omega])
        header = "  Frequency (THz)      "
    full_data_array = np.append(full_data_array, np.array([T_omega]), 
                                axis=0)
    header += "    T(omega), "+(str(temperature)+" K").ljust(11)

ymin, ymax = round_plot_range(0, Tomega_max, clamp_min=0)
ax.set_title("LiF", size=text_sizes[2])
ax.set_xlabel("Frequency ("+unit+")", fontsize=text_sizes[1])
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
ax.set_xlim(0, omega_max)
ax.set_ylim(ymin, ymax)
ax.legend(handles=plot_handles, fontsize=13, loc="upper right")
ax.tick_params(axis='both', labelsize=text_sizes[0])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
fig.tight_layout()
fig.show()

fig.savefig(plots_folder+"LiF_Tomega_temps_"+inputs_string+".pdf")
np.savetxt(plots_folder+"/LiF_Tomega_temps_"+inputs_string+"_data.txt", 
           np.transpose(full_data_array), header=header)




#%% Plot of the CUB Brillouin zone with high-symmetry path

data_folder = "data/KTaO3"
calc = PhonopyCommensurateCalculation(
    data_folder+"/KTaO3_a3.99_E0.0z_super444.yaml",
    born_filename=data_folder+"/BORN_KTaO3_a3.99.txt")
path, path_labels = get_path_and_labels("CUB", False)
label_shifts =  [
                    [ 0.12,  0.00,  0.00],      # Gamma
                    [-0.15,  0.00,  0.00],      # X
                    [ 0.12,  0.00,  0.00],      # M
                    [],
                    [ 0.00,  0.00,  0.05],      # R
                    [], [], []
                ]
label_style = dict(color="black", fontsize=14, horizontalalignment='center', 
                   verticalalignment='center_baseline')
reciprocal_lattice_vectors = None
quiver_labels = ["$\\bm{b}_1$", "$\\bm{b}_2$", "$\\bm{b}_3$"]
quiver_plot = np.empty((3, 3, 3))
# Data for the reciprocal lattice vectors:
quiver_plot[0:3,:,0] = 0.5*np.eye(3)
quiver_plot[0:3,:,1] = 0.2*np.eye(3)
quiver_plot[0:3,:,2] = 0.8*np.eye(3)

save_filename =  plots_folder+"CUB1BZ_high"
view_angles = (18, 26, 0)
save_bbox = [[1.5, 0.6], [5.15, 4.65]]
calc.plot_Brillouin(path=path, path_labels=path_labels, view_angles=view_angles,
                    label_shifts=label_shifts, quiver_labels=quiver_labels, 
                    quiver_plot=quiver_plot, save_filename=save_filename, 
                    reciprocal_lattice_vectors=reciprocal_lattice_vectors, 
                    label_style=label_style, save_bbox_extents=save_bbox)

#%% Plot of the CUB Brillouin zone with low-symmetry path

data_folder = "data/KTaO3"
calc = PhonopyCommensurateCalculation(
    data_folder+"/KTaO3_a3.99_E0.0z_super444.yaml",
    born_filename=data_folder+"/BORN_KTaO3_a3.99.txt")
path, path_labels = get_path_and_labels("CUB", True)
label_shifts =  [
                    [ 0.12,  0.00,  0.00],      # Gamma
                    [-0.15,  0.00,  0.00],      # X
                    [ 0.12,  0.00,  0.00],      # M
                    [],
                    [ 0.12,  0.00,  0.00],      # X1
                    [ 0.00,  0.00,  0.07],      # M1
                    [ 0.12,  0.00,  0.00],      # R
                    [], [], [], [], []
                ]
label_style = dict(color="black", fontsize=14, horizontalalignment='center', 
                   verticalalignment='center_baseline')
reciprocal_lattice_vectors = None
quiver_labels = ["$\\bm{b}_1$", "$\\bm{b}_2$", 
                 "$\\bm{b}_3, \\bm{\\mathcal{E}}$"]
quiver_plot = np.empty((3, 3, 3))
# Data for the reciprocal lattice vectors:
quiver_plot[0:3,:,0] = 0.5*np.eye(3)
quiver_plot[0:3,:,1] = 0.2*np.eye(3)
quiver_plot[0:3,:,2] = 0.8*np.eye(3)

save_filename = plots_folder+"CUB1BZ_low"
view_angles = (18, 26, 0)
save_bbox = [[1.5, 0.6], [5.15, 4.65]]


calc.plot_Brillouin(path=path, path_labels=path_labels, view_angles=view_angles,
                    label_shifts=label_shifts, quiver_labels=quiver_labels, 
                    quiver_plot=quiver_plot, save_filename=save_filename, 
                    reciprocal_lattice_vectors=reciprocal_lattice_vectors, 
                    label_style=label_style, save_bbox_extents=save_bbox)

#%% Plot of T(omega) convergence in KTaO3

unit="THz"
supercell_sizes = [2,4]
if recalculate:
    material_name = "KTaO3"
    moments = np.array([-0.5, -1.0, -1.5])
    a_str = "_a3.99"
    Efield = 0.005
    data_folder = "data/KTaO3"
    born_file = data_folder+"/BORN_"+material_name+a_str+".txt"
    common_str = data_folder+"/"+material_name+a_str+"_"
    print("Started KTaO3 T(omega) calculations at T=0K")
    print("---------------------------------------------")
    for index, supercell_size in enumerate(supercell_sizes):
        supercell_string = "super"+3*str(supercell_size)
        Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
        Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
        Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
        Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                            np.array([-Efield,0.00,Efield]), 
                            born_filename=born_file, take_imag=True)
        savedata_filename = "results/KTaO3/"+supercell_string+"/qmesh"\
            +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str+"_T0"
        results = Ycalc.calculate_Tomega(
            q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit, 
            sigma=recalc_sigma, include_nac="Gonze", moments=moments, 
            q_split_levels=2, parallel_jobs=recalc_parallel_jobs, 
            savedata_filename = savedata_filename, temperature=0)
        print("Finished "+str(index+1)+" of "+str(len(supercell_sizes))\
              +" calculations")
    print("\n")
omega_max = 0
Tomega_max = 0
fig, ax = plt.subplots()
plot_handles = []
inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a3.99"
for index, supercell_size in enumerate(supercell_sizes):
    supercell_string = "super"+str(supercell_size)*3  # e.g. super444
    supercell_string2 = "x".join(str(supercell_size)*3)  # e.g. 4x4x4
    results = TomegaResults("results/KTaO3/"+supercell_string+"/"+\
                            inputs_string+"_T0.npz")
    data = results.to_dict()
    omega = data["omega"]
    T_omega = data["Tomega"]
    if index == 0:
        full_data_array = np.array([omega])
        header = "  Frequency (THz)      "
    full_data_array = np.append(full_data_array, np.array([T_omega]), axis=0)
    header += "    T(omega), "+(supercell_string2).ljust(11)
    omega_max = max(omega[-1], omega_max)   
    Tomega_max = max(np.max(T_omega), Tomega_max) 
    mesh_str = "$"+str(supercell_size)+"\\times"\
        +str(supercell_size)+"\\times"+str(supercell_size)+"$"
    plot_handle, = ax.plot(omega, T_omega, label = mesh_str)
    plot_handles.append(plot_handle)
    print("Moments of T(omega) for "+supercell_string+":")
    print(data['Tmoments'])

_, Tomega_max_scale = round_plot_range(0, Tomega_max, clamp_min = 0)
    
ax.set_title("KTaO3", size=text_sizes[2])
ax.set_xlabel("Frequency ("+unit+")", fontsize=text_sizes[1])
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
ax.set_xlim(0, omega_max)
ax.set_ylim(0, Tomega_max_scale)
ax.legend(handles=plot_handles, fontsize=text_sizes[0], loc="upper right")
ax.tick_params(axis='both', labelsize=text_sizes[0])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
fig.tight_layout()
fig.show()

fig.savefig(plots_folder+"KTaO3_Tomega_conv_"+inputs_string+".pdf")

np.savetxt(plots_folder+"/KTaO3_Tomega_conv_"+inputs_string+"_data.txt", 
           np.transpose(full_data_array),
           header="  Frequency (THz)      "+\
                "    T(omega), 2x2x2      "+\
                "    T(omega), 4x4x4      ")

#%% Plot of strength of |Y_{nu1,nu2}(q)|^2 in KTaO3

path, path_labels = get_path_and_labels("CUB", True)
data_folder = "data/KTaO3"
Efield = 0.005
born_file = data_folder+"/BORN_KTaO3_a3.99.txt"
Eminus_file = data_folder+"/KTaO3_a3.99_E-"+str(Efield)+"z_super444.yaml"
Ezero_file = data_folder+"/KTaO3_a3.99_E0.0z_super444.yaml"
Eplus_file = data_folder+"/KTaO3_a3.99_E"+str(Efield)+"z_super444.yaml"
savefig_filename = save_filename=plots_folder+"Yplots_KTaO3_super444_a3.99"
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for KTaO3"
Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                      np.array([-Efield,0.00,Efield]), born_filename=born_file,
                      take_imag=True)
plot_data = Ycalc.plotY(path, path_labels, include_nac="Gonze", npoints=101,
                        num_markers=151,
                        degenerate_cutoff=1e-3, subplots=None,
                        save_filename=savefig_filename,
                        plot_range=(0,30), title2=title_string,
                        text_sizes=text_sizes)
Y2_max = plot_data[0]
Y2_sum_max = plot_data[1]
print("Maximum value of |Y|² on the individual plots: "+str(Y2_max)+"Å²")
print("Maximum value of |Y|² on the summed plot: "+str(Y2_sum_max)+"Å²")

#%% Plot of LATO contributions to T(omega) in KTaO3

inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a3.99"
results = TomegaResults("results/KTaO3/super444/"+inputs_string+"_T0.npz")
data = results.to_dict()
results.save_txt(plots_folder+"KTaO3_Tomega_LATO_"+inputs_string+"_data",
                 plots_folder+"KTaO3_Tmoments_LATO_"+inputs_string+"_data")
fig, axes, _ = results.plot(title="KTaO3", text_sizes=text_sizes)
# Set scientific notation on the main axis:
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
# Rename the text on the inset:
axes[1].set_ylabel("Relative energy contribution", fontsize=text_sizes[0])
fig.tight_layout()
fig.show()
fig.savefig(plots_folder+"KTaO3_Tomega_LATO_"+inputs_string+".pdf")

#%% Plot of T(omega) for different temperatures in KTaO3

unit="THz"
temperatures = [0,50,100,150,200,250,300]
inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a3.99"
if recalculate:
    material_name = "KTaO3"
    moments = np.array([-0.5, -1.0, -1.5])
    a_str = "_a3.99"
    Efield = 0.005
    data_folder = "data/KTaO3"
    born_file = data_folder+"/BORN_"+material_name+a_str+".txt"
    common_str = data_folder+"/"+material_name+a_str+"_"
    Eminus_file = common_str+"E-"+str(Efield)+"z_super444.yaml"
    Ezero_file = common_str+"E0.0z_super444.yaml"
    Eplus_file = common_str+"E"+str(Efield)+"z_super444.yaml"
    Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                        np.array([-Efield,0.00,Efield]), 
                        born_filename=born_file, take_imag=True)
    print("Started KTaO3 T(omega) temperature calculations for 4x4x4 supercell")
    print("-------------------------------------------------------------------")
    for index, temperature in enumerate(temperatures):
        savedata_filename = "results/KTaO3/super444/qmesh"\
            +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str\
            +"_T"+str(temperature)
        results = Ycalc.calculate_Tomega(
            q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit, 
            sigma=recalc_sigma, include_nac="Gonze", moments=moments, 
            q_split_levels=2, parallel_jobs=recalc_parallel_jobs, 
            savedata_filename = savedata_filename, temperature=temperature)
        print("Finished "+str(index+1)+" of "+str(len(temperatures))\
              +" calculations")
    print("\n")
omega_max = 0
Tomega_max = 0
fig, ax = plt.subplots()
plot_handles = []
for index, temperature in enumerate(temperatures):
    temp_str = "T"+str(temperature)
    data = np.load("results/KTaO3/super444/"+inputs_string+"_"+temp_str+".npz")
    omega = data["omega"]
    T_omega = data["Tomega"]  
    omega_max = max(omega[-1], omega_max)   
    Tomega_max = max(np.max(T_omega), Tomega_max) 
    legend_str = "$T = "+str(temperature)+"$K"
    plot_handle, = ax.plot(omega, T_omega, label = legend_str)
    plot_handles.append(plot_handle)
    if index == 0:
        full_data_array = np.array([omega])
        header = "  Frequency (THz)      "
    full_data_array = np.append(full_data_array, np.array([T_omega]), 
                                axis=0)
    header += "    T(omega), "+(str(temperature)+" K").ljust(11)

ymin, ymax = round_plot_range(0, Tomega_max, clamp_min=0)
ax.set_title("KTaO3", size=text_sizes[2])
ax.set_xlabel("Frequency ("+unit+")", fontsize=text_sizes[1])
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
ax.set_xlim(0, omega_max)
ax.set_ylim(ymin, ymax)
ax.legend(handles=plot_handles, fontsize=13, loc="upper right")
ax.tick_params(axis='both', labelsize=text_sizes[0])
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
fig.tight_layout()
fig.show()

fig.savefig(plots_folder+"KTaO3_Tomega_temps_"+inputs_string+".pdf")
np.savetxt(plots_folder+"/KTaO3_Tomega_temps_"+inputs_string+"_data.txt", 
           np.transpose(full_data_array), header=header)
