import os
import numpy as np
import scipy
import scipy.integrate as intgr
import matplotlib as mpl
import matplotlib.pyplot as plt
from phonopyReaders import PhonopyCommensurateCalculation, YCalculation, \
    TomegaResults, round_plot_range
from pathsLabels import get_path_and_labels

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble']="\\usepackage{bm}"
# mpl.rcParams['text.latex.preamble']="\\usepackage{amsmath}"
np.set_printoptions(precision=6, suppress=True)

### Generate all the plots for alpha-CsPbI3
plots_folder = "mobility_plots/"

## Brillouin-zone plot
#%% Plot of the CUB Brillouin zone with low-symmetry path
data_folder = "data/CsPbI3"
params_string = "a6.276_EDIFF1e-08_disp0.005"
calc = PhonopyCommensurateCalculation(
    data_folder+"/CsPbI3_a6.276_EDIFF1e-08_disp0.005_E0.0z_super222.yaml",
    born_filename=data_folder+"/BORN_CsPbI3_"+params_string+".txt")
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
# quiver_labels = ["$b_1$", "$b_2$", 
#                  "$b_3, E$"]
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

## Plot of Y²
imaginary_cutoff = 0.1
Efield = 0.0025
data_folder = "data/CsPbI3"
path, path_labels = get_path_and_labels("CUB", True)
supercell_string = "super222"
params_string = "a6.276_EDIFF1e-08_disp0.005"
cutoff_str = "_cutoff"+str(imaginary_cutoff).replace(".","")

born_file = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
common_str = data_folder+"/CsPbI3_"+params_string+"_"
Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
savefig_filename = plots_folder+"Yplots/CsPbI3_"+supercell_string+cutoff_str+"_"+params_string
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for $\\alpha$-CsPbI$_3$"
Y2_unit = "Å²"
Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                    np.array([-Efield,0.00,Efield]), born_filename=born_file, take_imag=True)
plot_data = Ycalc.plotY(path, path_labels, include_nac="Gonze", npoints=101, num_markers=151,
                        degenerate_cutoff=1e-3, subplots=None, save_filename=savefig_filename,
                        plot_range=(-1,4), title2=title_string, imaginary_cutoff=imaginary_cutoff,
                        Y2_sum_norm_value=1e-4, Y2_norm_value=1e-4)
Y2_max = plot_data[0]
Y2_sum_max = plot_data[1]
print("Maximum value of |Y|² on the individual plots: "+str(Y2_max)+" "+Y2_unit)
print("Maximum value of |Y|² on the summed plot: "+str(Y2_sum_max)+" "+Y2_unit)
fig_handles = plot_data[2]
ax_handles = plot_data[3]
plot_handles = plot_data[4]
fig = fig_handles[-1]
ax = ax_handles[-1]
for index in range(len(fig_handles)-1):
    plt.close(fig_handles[index])
ax2 = ax.twinx()
color = "#0000FF"
ax2.tick_params(axis='y', labelcolor=color, labelsize=13)
limits_F = np.array(ax.get_ylim())
limits_K = limits_F*4.135667696e-3/8.61733262e-5  # convert frequencies in THz to temperatures in K
ax2.set_ylim(limits_K)
ax2.set_yticks([0, 50, 100, 150], labels=["0 K", "50 K", "100 K", "150 K"])
ytick_values = ax2.get_yticks()
ytick_labels = list(map(lambda x: str(int(x))+" K", ytick_values))
ax2.set_yticklabels(ytick_labels)
ax2.set_ylabel("$\\frac{\\hbar \\omega_{\\mathbf{q},\\nu}}{k_B}$", color=color, fontsize=18,
               rotation=0, ha='left', va='center')
fig.tight_layout()
fig.savefig(savefig_filename+".pdf")

## Plot of T(omega)
data_folder = "choose_data/CsPbI3_alpha/GWpotcars_PBEex"
params_string = "a6.276_EDIFF1e-08_disp0.005"
Ezero_file = data_folder+"/CsPbI3_"+params_string+"_E0.0z_super222.yaml"
temperatures = [0,100,200,300]
born_file = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
my_calc = PhonopyCommensurateCalculation(Ezero_file, born_filename=born_file)
freqs, polarities = my_calc.get_polarities(lebedev_grid=21, averaged=True, unit="THz")
relevant_modes = np.nonzero(polarities > 1e-3)  # Find the modes with nonzero mode polarities
omega_nus = freqs[relevant_modes]
p2_nus = polarities[relevant_modes]**2
print("LO phonon frequencies (THz): " + str(omega_nus))
print("Squared mode polarities ((a.m.u.)^-1): " + str(p2_nus))
delta_areas = (1.602176634e-19)**2/(2*8.8541878188e-12*(6.24e-10)**3)*(p2_nus/1.66053906892e-27)/(omega_nus*(2*np.pi*1e12)**2)
annotate_ys = np.array([0.24, 0.22, 0.15])
arrowprops = dict(arrowstyle="->, head_width=0.1", relpos=(0., 0.5))
direc = "results/CsPbI3/super222_cutoff01"
fig, ax = plt.subplots()
plot_handles = []
for temperature in temperatures:
    calc = TomegaResults(direc+"/qmesh128_sigma0.025_"+params_string+"_T"+str(int(temperature))+".npz")
    plot_handle, = ax.plot(calc.omega, calc.Tomega, label="$T="+str(temperature)+"$ K")
    plot_handles.append(plot_handle)
ax.vlines(omega_nus, 0, 0.25, color="black", linewidth=0.7)
for index, (omega, area, y) in enumerate(zip(omega_nus, delta_areas, annotate_ys)):
    areastr = '{0:.1f}'.format(area)
    p2str_full = "Area: $"+areastr+"$ THz"
    ax.annotate(p2str_full, xy=(omega, y), xytext=(omega+0.7, y),
                ha="left", va="center",
                arrowprops=arrowprops, fontsize=12)
    
    
ax.set_xlim([0,8])
ax.set_ylim([0,0.25])
ax.set_xlabel("Frequency (THz)", fontsize=16)
ax.set_ylabel("$\\mathcal{P}(\\omega)$", fontsize=16)
ax.set_title("Electron-phonon spectral function", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(handles=plot_handles, loc="upper right", fontsize=13)
fig.tight_layout()
fig.savefig(plots_folder+"SpectralFunction.pdf")
plt.show()