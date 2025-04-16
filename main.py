"""
Example usage that generates all figures in arXiv:2412.09456
and arXiv:2412.09470

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
recalculate = False  # Warning: recalculating data takes several days
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
inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a4.004"
# if recalculate:
#     material_name = "LiF"
#     moments = np.array([-0.5, -1.0, -1.5])
#     a_str = "_a4.004"
#     Efield = 0.01
#     data_folder = "data/LiF/Efields"
#     born_file = data_folder+"/BORN_"+material_name+a_str
#     common_str = data_folder+"/"+material_name+a_str+"_"
#     for supercell_size in supercell_sizes:
#         supercell_string = "super"+3*str(supercell_size)
#         Eminus_file = common_str+"E-"+str(Efield)+"_"+supercell_string+".yaml"
#         Ezero_file = common_str+"E0.0_"+supercell_string+".yaml"
#         Eplus_file = common_str+"E"+str(Efield)+"_"+supercell_string+".yaml"
#         Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
#                             np.array([-Efield,0.00,Efield]), 
#                             born_filename=born_file, take_imag=True)
#         plot_title = "LiF "+supercell_string
#         savedata_filename = "results/LiF/"+supercell_string+"/qmesh"\
#             +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str
#         savefig_filename = None
#         savetxt_filename = None
#         Ycalc.calculate_Tomega(
#             q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit, 
#             sigma=recalc_sigma, include_nac="Gonze", moments=moments, 
#             q_split_levels=2, parallel_jobs=recalc_parallel_jobs, 
#             savedata_filename = savedata_filename, 
#             savefigures_filename = savefig_filename, 
#             savetxt_filename = savetxt_filename, title=plot_title)
omega_max = 0
Tomega_max = 0
fig, ax = plt.subplots()
plot_handles = []
for index, supercell_size in enumerate(supercell_sizes):
    supercell_string = "super"+str(supercell_size)*3
    results = TomegaResults("results/LiF/"+supercell_string+"/"+inputs_string+"_T0.npz")
    data = results.to_dict()
    omega = data["omega"]
    T_omega = data["Tomega"]
    if index == 0:
        full_data_array = np.array([omega, T_omega])
    else:
        full_data_array = np.append(full_data_array, np.array([T_omega]), 
                                    axis=0)  
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

np.savetxt("plots/LiF_Tomega_conv_"+inputs_string+"_data.txt", 
           np.transpose(full_data_array),
           header="  Frequency (THz)      "+\
                "    T(omega), 2x2x2      "+\
                "    T(omega), 3x3x3      "+\
                "    T(omega), 4x4x4      "+\
                "    T(omega), 5x5x5      "+\
                "    T(omega), 6x6x6      ")

# #%% Plot of phonon bands and LATO weights in LiF

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

# #%% Plot of strength of |Y_{nu1,nu2}(q)|^2 in LiF

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

# #%% Plot of LATO contributions to T(omega) in LiF

# if recalculate:
    # unit="THz"
    # material_name = "LiF"
    # moments = np.array([-0.5, -1.0, -1.5])
    # supercell_string = "super666"
    # a_str = "_a4.004"
    # Efield = 0.01
    # data_folder = "data/LiF/Efields"
    # born_file = data_folder+"/BORN_"+material_name+a_str
    # common_str = data_folder+"/"+material_name+a_str+"_"
    # Eminus_file = common_str+"E-"+str(Efield)+"_"+supercell_string+".yaml"
    # Ezero_file = common_str+"E0.0_"+supercell_string+".yaml"
    # Eplus_file = common_str+"E"+str(Efield)+"_"+supercell_string+".yaml"
    # Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
    #                     np.array([-Efield,0.00,Efield]), 
    #                     born_filename=born_file, take_imag=True)
    # plot_title = "LiF"
    # savedata_filename = "results/LiF/"+supercell_string+"/qmesh"\
    #     +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str
    # savefig_filename = plots_folder+"LiF_Tomega_LATO_"+inputs_string
    # savetxt_filename = plots_folder+"LiF_Tomega_LATO_"+inputs_string+"_data"
    # Ycalc.calculate_Tomega(
    #     q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit,
    #     sigma=recalc_sigma, include_nac="Gonze", moments=moments,
    #     q_split_levels=2, parallel_jobs=recalc_parallel_jobs,
    #     savedata_filename = savedata_filename, 
    #     savefigures_filename = savefig_filename,
    #     savetxt_filename = savetxt_filename,
    #     title=plot_title)

inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a4.004"
results = TomegaResults("results/LiF/super666/"+inputs_string+"_T0.npz")
data = results.to_dict()
results.save_txt("plots/LiF_Tomega_LATO_"+inputs_string+"_data",
                 "plots/LiF_Tmoments_LATO_"+inputs_string+"_data")
fig, axes, _ = results.plot(title="LiF", text_sizes=text_sizes)
# Set scientific notation on the main axis:
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
# Rename the text on the inset:
axes[1].set_ylabel("Relative energy contribution", fontsize=text_sizes[0])
fig.tight_layout()
fig.show()
fig.savefig(plots_folder+"LiF_Tomega_LATO_"+inputs_string+".pdf")





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

# #%% Plot of strength of |Y_{nu1,nu2}(q)|^2 in KTaO3

# path, path_labels = get_path_and_labels("CUB", True)
# data_folder = "data/KTaO3/Efields"
# Efield = 0.005
# born_file = data_folder+"/BORN_KTaO3_a3.99"
# Eminus_file = data_folder+"/KTaO3_a3.99_E-"+str(Efield)+"_super444.yaml"
# Ezero_file = data_folder+"/KTaO3_a3.99_E0.0_super444.yaml"
# Eplus_file = data_folder+"/KTaO3_a3.99_E"+str(Efield)+"_super444.yaml"
# savefig_filename = save_filename=plots_folder+"Yplots_KTaO3_super444_a3.99"
# title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for KTaO$_3$"
# Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
#                       np.array([-Efield,0.00,Efield]), born_filename=born_file,
#                       take_imag=True)
# plot_data = Ycalc.plotY(path, path_labels, include_nac="Gonze", npoints=101, 
#                         num_markers=151, degenerate_cutoff=1e-3, subplots=None,
#                         save_filename=savefig_filename, plot_range=(0,30),
#                         title2=title_string, text_sizes=text_sizes)
# Y2_max = plot_data[0]
# Y2_sum_max = plot_data[1]
# print("Maximum value of |Y|² on the individual plots: "+str(Y2_max)+"Å²")
# print("Maximum value of |Y|² on the summed plot: "+str(Y2_sum_max)+"Å²")

# #%% Plot of LATO contributions to T(omega) in KTaO3

# colorTA = (0.8, 0.5, 0.5)
# colorLA = (0.4, 0.7, 0.4)
# colorTO = (0.4, 0.4, 0.8)
# colorLO = (0.3, 0.3, 0.3)
# hatchplot_styles = [
#     dict(color1=colorTA, color2=colorTA, label_shift = -0.02, hatch='//'),
#     dict(color1=colorTA, color2=colorLA, label_shift = 0.02, hatch='//'),
#     dict(color1=colorTA, color2=colorTO, label_shift = 0, hatch='\\\\'),
#     dict(color1=colorTA, color2=colorLO, label_shift = -0.005, hatch='//'),
#     dict(color1=colorLA, color2=colorLA, label_shift = 0, hatch='//'),
#     dict(color1=colorLA, color2=colorTO, label_shift = +0.005, hatch='//'),
#     dict(color1=colorLA, color2=colorLO, label_shift = 0, hatch='\\\\'),
#     dict(color1=colorTO, color2=colorTO, label_shift = 0, hatch='//'),
#     dict(color1=colorTO, color2=colorLO, label_shift = 0, hatch='//'),
#     dict(color1=colorLO, color2=colorLO, label_shift = 0, hatch='//'),
# ]

# unit="THz"
# inputs_string = "qmesh"+str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+"_a3.99"

# if recalculate:
#     material_name = "KTaO3"
#     moments = np.array([-0.5, -1.0, -1.5])
#     supercell_string = "super444"
#     a_str = "_a3.99"
#     Efield = 0.005
#     data_folder = "data/KTaO3/Efields"
#     born_file = data_folder+"/BORN_"+material_name+a_str
#     common_str = data_folder+"/"+material_name+a_str+"_"
#     Eminus_file = common_str+"E-"+str(Efield)+"_"+supercell_string+".yaml"
#     Ezero_file = common_str+"E0.0_"+supercell_string+".yaml"
#     Eplus_file = common_str+"E"+str(Efield)+"_"+supercell_string+".yaml"
#     Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
#                         np.array([-Efield,0.00,Efield]), 
#                         born_filename=born_file, take_imag=True)
#     plot_title = "KTaO$_3$"
#     savedata_filename = "results/KTaO3/"+supercell_string+"/qmesh"\
#         +str(recalc_qmesh)+"_sigma"+str(recalc_sigma)+a_str
#     savefig_filename = plots_folder+"KTaO3_Tomega_LATO_"+inputs_string
#     savetxt_filename = plots_folder+"KTaO3_Tomega_LATO_"+inputs_string+"_data"
#     Ycalc.calculate_Tomega(
#         q_mesh_size=recalc_qmesh, num_omegas=1001, unit=unit,
#         sigma=recalc_sigma, include_nac="Gonze", moments=moments,
#         q_split_levels=2, parallel_jobs=recalc_parallel_jobs,
#         savedata_filename = savedata_filename, 
#         savefigures_filename = savefig_filename,
#         savetxt_filename = savetxt_filename, title=plot_title)
# else:
#     data = np.load("results/KTaO3/super444/qmesh128_sigma0.1_a3.99.npz")
#     omega = data["omega"]
#     T_omega = data["Tomega"]  
#     labels = ["TA", "LA", "TO", "LO"]
#     name = "LATO"
#     partials = [[0],[1],[2],[3]]
#     Tomega_res = data['Tomega_LATO']
#     Tmoments_res = data['Tmoments_LATO']
        
#     num_partials = len(partials)
#     num_contributions = int(num_partials*(num_partials+1)/2)
#     T_contributions = np.zeros((num_contributions, len(omega)))
#     T_moments_contributions = np.zeros((num_contributions, len(Tmoments_res)))
#     contribution_labels = []
#     count = 0
#     for index1 in range(0, num_partials):
#         T_contributions[count] = Tomega_res[:, index1, index1]
#         T_moments_contributions[count] = Tmoments_res[:, index1, index1]
#         contribution_labels.append(labels[index1]+"-"+labels[index1])
#         count += 1
#         for index2 in range(index1+1, num_partials):
#             T_contributions[count] = Tomega_res[:, index1, index2] \
#                 + Tomega_res[:, index2, index1]
#             T_moments_contributions[count] = Tmoments_res[:, index1, index2] \
#                 + Tmoments_res[:, index2, index1]
#             contribution_labels.append(labels[index1]+"-"+labels[index2])
#             count += 1
#     T_mhalf_relative = T_moments_contributions[:,0] \
#         / np.sum(T_moments_contributions[:,0])

#     fig, ax = plt.subplots()
#     ax2 = ax.inset_axes([0.79,0.15,0.1,0.8])
#     total_stack = np.zeros_like(omega)
#     total_stack2 = 0
#     plot_handles = []
#     plot_handles2 = []
#     for index, (data, T_mhalf, style, label) \
#         in enumerate(zip(T_contributions, T_mhalf_relative,
#                         hatchplot_styles, contribution_labels)):
#         plot_handle = \
#             ax.fill_between(omega, total_stack + data, y2=total_stack, 
#                             zorder=-index, linewidth=0, label=label,
#                             hatch=style['hatch'], fc=style['color1'], 
#                             edgecolor=style['color2'])
#         ax.plot(omega, total_stack + data, zorder=0, color='k', linewidth=0.5)
#         plot_handle2 = ax2.bar(0.0, T_mhalf, 1.0, label=label, zorder=-index, 
#                             bottom=total_stack2, linewidth=0,
#                             hatch=style['hatch'], fc=style['color1'], 
#                             edgecolor=style['color2'])
#         ax2.plot(np.array([-0.5,0.5]), np.tile(total_stack2 + T_mhalf, 2), 
#                 zorder=0, color='k', linewidth=0.5)
#         ax2.text(0.51, total_stack2 + 0.5*T_mhalf + style["label_shift"], label, 
#                 fontsize=12, horizontalalignment="left", 
#                 verticalalignment="center_baseline",)
#         plot_handles.append(plot_handle)
#         plot_handles2.append(plot_handle2)
#         total_stack += data
#         total_stack2 += T_mhalf
#     plot_handle_total, = ax.plot(omega, T_omega, color="black", label="Total")
#     plot_handles.append(plot_handle_total)

#     ax.set_title("KTaO$_3$", fontsize=text_sizes[2])
#     ax.set_xlabel("Frequency ("+unit+")", fontsize=text_sizes[1])
#     ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
#     # ax.legend(handles=plot_handles[::-1], fontsize=13)
#     ax.set_xlim(0, 55)
#     ax.set_ylim(0, 0.006)
#     ax.tick_params(axis='both', labelsize=text_sizes[0])
#     ax2.set_ylabel("Relative energy contribution", fontsize=12)
#     ax2.tick_params(left=False, labelleft=False, bottom=False, 
#                     labelbottom=False)
#     ax2.set_xlim(-0.5, 0.5)
#     ax2.set_ylim( 0.0, 1)
#     ax2.tick_params(axis='both', labelsize=text_sizes[0])
#     fig.tight_layout()
#     fig.show()
#     fig.savefig(plots_folder+"KTaO3_Tomega_LATO_qmesh128_sigma0.1_a3.99.pdf")

#     full_data_array = np.transpose(np.append(np.array([omega, T_omega]), 
#                                              T_contributions, axis=0))
#     np.savetxt("plots/KTaO3_Tomega_LATO_qmesh128_sigma0.1_a3.99_data.txt", 
#             full_data_array,
#             header="  Frequency (THz)      "+\
#                     "    T(omega), total      "+\
#                     "    T(omega), TA-TA      "+\
#                     "    T(omega), TA-LA      "+\
#                     "    T(omega), TA-TO      "+\
#                     "    T(omega), TA-LO      "+\
#                     "    T(omega), LA-LA      "+\
#                     "    T(omega), LA-TO      "+\
#                     "    T(omega), LA-LO      "+\
#                     "    T(omega), TO-TO      "+\
#                     "    T(omega), TO-LO      "+\
#                     "    T(omega), LO-LO      ")
