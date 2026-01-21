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
mpl.rcParams['text.latex.preamble']="\\usepackage{amsmath}"
np.set_printoptions(precision=6, suppress=True)

## Plot of Y² for alpha-CsPbI3
imaginary_cutoff = 0.1
Efield = 0.0025
data_folder = "choose_data/CsPbI3_alpha/EDIFF1e-7_disp0001"
path, path_labels = get_path_and_labels("CUB", True)
supercell_string = "super222"
params_string = "a6.24"
cutoff_str = "_cutoff"+str(imaginary_cutoff).replace(".","")

born_file = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
common_str = data_folder+"/CsPbI3_"+params_string+"_"
Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
savefig_filename = "mobility_plots/Yplots/CsPbI3_"+supercell_string+cutoff_str+"_"+params_string
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
# ax2.yticks([0, 50, 100, 150], labels=["0 K", "50 K", "100 K", "150 K"])
ytick_values = ax2.get_yticks()
ytick_labels = list(map(lambda x: str(int(x))+" K", ytick_values))
ax2.set_yticklabels(ytick_labels)
ax2.set_ylabel("$\\frac{\\hbar \\omega_{\\mathbf{q},\\nu}}{k_B}$", color=color, fontsize=18,
               rotation=0, ha='left', va='center')
fig.tight_layout()
fig.savefig(savefig_filename+".pdf")

## Same plot for LiF
imaginary_cutoff = 0.1
Efield = 0.01
data_folder = "data/LiF"
path, path_labels = get_path_and_labels("FCC", True)
supercell_string = "super444"
params_string = "a4.004"

born_file = data_folder+"/BORN_LiF_"+params_string+".txt"
common_str = data_folder+"/LiF_"+params_string+"_"
Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
savefig_filename = "plots/Yplots/LiF_"+supercell_string+"_"+params_string
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for LiF"
Y2_unit = "Å²"
Ycalc = YCalculation([Eminus_file,Ezero_file,Eplus_file], 
                    np.array([-Efield,0.00,Efield]), born_filename=born_file, take_imag=True)
plot_data = Ycalc.plotY(path, path_labels, include_nac="Gonze", npoints=101, num_markers=151,
                        degenerate_cutoff=1e-3, subplots=None, save_filename=savefig_filename,
                        plot_range=(0,20), title2=title_string,
                        Y2_sum_norm_value=1.5e-6, Y2_norm_value=1.5e-6)
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
# ax2.yticks([0, 50, 100, 150], labels=["0 K", "50 K", "100 K", "150 K"])
ytick_values = ax2.get_yticks()
ytick_labels = list(map(lambda x: str(int(x))+" K", ytick_values))
ax2.set_yticklabels(ytick_labels)
ax2.set_ylabel("$\\frac{\\hbar \\omega_{\\mathbf{q},\\nu}}{k_B}$", color=color, fontsize=18,
               rotation=0, ha='left', va='center')
fig.tight_layout()
fig.savefig(savefig_filename+".pdf")

## Plot of T(omega)
temperatures = [0,100,200,300]
omega_nus = np.array([0.626815, 0.946313, 3.901953])  # Model phonon frequencies, in THz
p2_nus = np.array([0.002353, 0.006116, 0.307695])  # Squared mode polarities, in (a.m.u.)^-1
delta_areas = (1.602176634e-19)**2/(2*8.8541878188e-12*(6.24e-10)**3)*(p2_nus/1.66053906892e-27)/(omega_nus*(2*np.pi*1e12)**2)
annotate_ys = np.array([0.33, 0.31, 0.15])
arrowprops=dict(arrowstyle="->, head_width=0.1", relpos=(0., 0.5))
# direc = "choose_results/CsPbI3_alpha/EDIFF1e-7_disp0001/super222_cutoff01"
direc = "choose_results/CsPbI3_alpha/disp0001/super222_cutoff01"
fig, ax = plt.subplots()
plot_handles = []
for temperature in temperatures:
    calc = TomegaResults(direc+"/qmesh64_sigma0.05_a6.24_T"+str(int(temperature))+".npz")
    plot_handle, = ax.plot(calc.omega, calc.Tomega, label="$T="+str(temperature)+"$ K")
    plot_handles.append(plot_handle)
ax.vlines(omega_nus, 0, 0.35, color="black", linewidth=0.7)
# for index, (omega, p2, y) in enumerate(zip(omega_nus, p2_nus, annotate_ys)):
    # p2str = '{0:.3f}'.format(p2)
    # p2str_full = "$|p_{\\text{LO},"+str(index+1)+"}|^2 = "+p2str+"$ (a.m.u.)$^{-1}$"
    # ax.annotate(p2str_full, xy=(omega, y), xytext=(omega+0.7, y),
    #             ha="left", va="center",
    #             arrowprops=arrowprops, fontsize=12)
for index, (omega, area, y) in enumerate(zip(omega_nus, delta_areas, annotate_ys)):
    areastr = '{0:.1f}'.format(area)
    p2str_full = "Area: $"+areastr+"$ THz"
    ax.annotate(p2str_full, xy=(omega, y), xytext=(omega+0.7, y),
                ha="left", va="center",
                arrowprops=arrowprops, fontsize=12)
    
    
ax.set_xlim([0,8])
ax.set_ylim([0,0.35])
ax.set_xlabel("Frequency (THz)", fontsize=16)
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=16)
ax.set_title("1-electron-2-phonon spectral function", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(handles=plot_handles, loc="upper right", fontsize=13)
fig.tight_layout()
fig.savefig("mobility_plots/SpectralFunction.pdf")
plt.show()