import os
import time
import numpy as np
import scipy
import scipy.integrate as intgr
import matplotlib as mpl
import matplotlib.pyplot as plt
from phonopyReaders import PhonopyCommensurateCalculation, YCalculation, \
    TomegaResults, round_plot_range, create_path
from pathsLabels import get_path_and_labels

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble']="\\usepackage{bm}\\usepackage{amsmath}"
np.set_printoptions(precision=6, suppress=True)

# Define nature constants:
e = 1.602176634e-19  # Elementary charge, in Coulomb
hbar = 6.62607015e-34/(2*np.pi)  #Reduced Planck constant, in J.s
epsvac = 8.8541878188e-12  # Vacuum permittivity, in C^2/(J.m)
mel = 9.1093837139e-31  # Electron mass, in kg
amu = 1.66053906892e-27  # Atomic mass unit, in kg
kB = 1.380649e-23  # Boltzmann constant in J/K

### MAIN ARTICLE PLOTS
recalculate = False  # If False, use precomputed data for the heaviest calculations
plots_folder = "mobility_plots/"
color_list = ["#FF4444", "#00BB00", "#4444BB", "#DD9922", "#BB44BB", "#888888"]

## === Brillouin-zone plot ===
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
## ===========================

## === Plot of Y² ===
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
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for CsPbI$_3$"
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
## ==================

## === Plot of T(omega) ===
data_folder = "data/CsPbI3"
params_string = "a6.276_EDIFF1e-08_disp0.005"
Ezero_file = data_folder+"/CsPbI3_"+params_string+"_E0.0z_super222.yaml"
q_mesh_size = 128
sigma = 0.025
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
for index, temperature in enumerate(temperatures):
    results_filename = direc+"/qmesh"+str(q_mesh_size)+"_sigma"+str(sigma)+"_"+params_string+"_T"+str(int(temperature))+".npz"
    if recalculate:  # Recalculate data from scratch
        print("Recalculating T(ω) at T="+str(int(temperature))+"K with σ="+str(sigma)
              +"THz and a "+str(q_mesh_size)+"x"+str(q_mesh_size)+"x"+str(q_mesh_size)+" q-mesh")
        start_time = time.time()
        Efields = [-0.0025, 0.0, 0.0025]  # units of V/Å
        yaml_filenames = [data_folder+"/CsPbI3_"+params_string+"_E"+str(Efield)+"z_super222.yaml" for Efield in Efields]
        born_filename = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
        calcY = YCalculation(yaml_filenames, Efields, born_filename, take_imag=True)
        results = calcY.calculate_Tomega(q_mesh_size = q_mesh_size, sigma = sigma, 
                                         include_nac="Gonze", temperature = temperature,
                                         q_split_levels = 2, parallel_jobs = 6, imaginary_cutoff = 0.1)
        results.save_npz(results_filename)
        print("Calculation time: "+str(time.time() - start_time)+"s")
    calc = TomegaResults(results_filename)
    plot_handle, = ax.plot(calc.omega, calc.Tomega, label="$T="+str(temperature)+"$ K", color = color_list[index])
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
fig.savefig(plots_folder+"spectral_function.pdf")
## ========================

## === Recalculate all T(omega) for the mobility ===
q_mesh_size = 32
sigma = 0.025
temperatures = np.arange(10, 510, 10)
cutoffs = [0.05, 0.1, 0.2]
params_string = "a6.276_EDIFF1e-08_disp0.005"
if recalculate:  # Recalculate data from scratch
    for imaginary_cutoff in cutoffs:
        for temperature in temperatures:
            cutoff_str = "_cutoff"+str(imaginary_cutoff).replace(".","")
            direc = "results/CsPbI3/super222"+cutoff_str
            results_filename = direc+"/qmesh"+str(q_mesh_size)+"_sigma"+str(sigma)+"_"+params_string+"_T"+str(int(temperature))+".npz"
            print("Recalculating T(ω) at T="+str(int(temperature))+"K with σ="+str(sigma)
                    +"THz and a "+str(q_mesh_size)+"x"+str(q_mesh_size)+"x"+str(q_mesh_size)+" q-mesh")
            start_time = time.time()
            Efields = [-0.0025, 0.0, 0.0025]  # units of V/Å
            yaml_filenames = ["data/CsPbI3/CsPbI3_"+params_string+"_E"+str(Efield)+"z_super222.yaml" for Efield in Efields]
            born_filename = "data/CsPbI3/BORN_CsPbI3_"+params_string+".txt"
            calcY = YCalculation(yaml_filenames, Efields, born_filename, take_imag=True)
            results = calcY.calculate_Tomega(q_mesh_size = q_mesh_size, sigma = sigma, 
                                             include_nac="Gonze", temperature = temperature,
                                             q_split_levels = 2, parallel_jobs = 6,
                                             imaginary_cutoff=imaginary_cutoff)
            results.save_npz(results_filename)
            print("Calculation time: "+str(time.time() - start_time)+"s")
## =================================================

## === Plot of the inverse lifetimes ===
## =====================================
data_folder = "choose_data/CsPbI3_alpha/GWpotcars_PBEex"
calc_params_string = "a6.276_EDIFF1e-08_disp0.005"
Ezero_file = data_folder+"/CsPbI3_"+calc_params_string+"_E0.0z_super222.yaml"
born_file = data_folder+"/BORN_CsPbI3_"+calc_params_string+".txt"
my_calc = PhonopyCommensurateCalculation(Ezero_file, born_filename=born_file)
freqs, polarities = my_calc.get_polarities(lebedev_grid=21, averaged=True, unit="meV")
relevant_modes = np.nonzero(polarities > 1e-3)  # Find the modes with nonzero mode polarities
omega_nus = freqs[relevant_modes]
p2_nus = polarities[relevant_modes]**2
epsinf = np.trace(my_calc.dielectric_tensor)/3 # High-frequency dielectric constant
params_string = "qmesh32_sigma0.025_"+calc_params_string
band_mass = 0.17*mel  # Electron band mass, in kg; taken from Poncé 2019, doi.org/10.1021/acsenergylett.8b02346
lattice_constant = 6.276  # CsPbI3 lattice constant, in Å
unit_cell_volume_SI = lattice_constant**3*1e-30  # Unit cell volume, in m^3
omega_nus_SI = omega_nus*1e-3*e/hbar  # Model phonon frequencies in rad/s
G2_nus_SI = p2_nus*(e**2/(epsvac*epsinf*unit_cell_volume_SI))**2*hbar/(2*omega_nus_SI) / 1.66053906892e-27 # Matrix element strengths in N^2

def phi(x, x0):
    """Auxiliary function to calculate the inverse lifetimes"""
    return np.sqrt(x0/x)*(np.arcsinh(np.sqrt(x/x0))/(np.exp(x0)-1)
                          + np.arccosh(np.maximum(1,np.sqrt(x/x0))) \
                            *(x>x0)/(1-np.exp(-x0)))

def tau_inv(xs, temperature):
    """Linear contribution to the inverse lifetimes"""

    # Use analytical formula for the linear contribution:
    X, T, W = np.meshgrid(xs, temperature, omega_nus_SI)
    return (2/hbar**2)*unit_cell_volume_SI/(4*np.pi)*np.sqrt(2*band_mass/hbar) \
        * np.sum(G2_nus_SI/np.sqrt(W)*phi(X, hbar*W/(kB*T)), 2)

def tau2_inv(xs, temperature, directory, params_string):
    """Nonlinear contribution to the inverse lifetimes"""

    # Load T(omega) at the correct temperature:
    calc = TomegaResults(directory+params_string+"_T"+str(int(temperature))+".npz")
    # Use analytical function in terms of T(omega) for the nonlinear contribution:
    x = calc.omega*(2*np.pi*1e12)
    W, X = np.meshgrid(x, xs)
    y = calc.Tomega*phi(X, hbar*W/(kB*temperature))/np.sqrt(W)
    return (2/hbar)*e**2/(4*np.pi*epsvac*epsinf**2)*np.sqrt(2*band_mass/hbar) \
        * intgr.simpson(y[:,1:], x[1:], axis=1)

lambda_max = 0.085
k_max = 2*np.pi*hbar/(lattice_constant*1e-10*np.sqrt(band_mass*kB))*lambda_max
ks = np.linspace(0, k_max, 201)[1:]
Ts = [10, 150, 300]
fig, ax = plt.subplots()
plot_handles = []
for index, T in enumerate(Ts):
    legend_entry_1 = "$T="+str(T)+"~K$, 1e1ph"
    legend_entry_12 = "$T="+str(T)+"~K$, 1e1ph+1e2ph"
    tau1_cont = tau_inv(ks**2/T, T)[0]
    tau2_cont = tau2_inv(ks**2/T, T, directory="results/CsPbI3/super222_cutoff01/", params_string=params_string)
    tau_invs = tau1_cont + tau2_cont
    color = color_list[index]
    plot_handle1, = ax.plot(ks, tau1_cont*1e-12, color=color, label=legend_entry_1)
    plot_handle2, = ax.plot(ks, tau_invs*1e-12, color=color, linestyle="dashed", label=legend_entry_12)
    plot_handles.append(plot_handle1)
    plot_handles.append(plot_handle2)
ax.set_ylim([0,450])
ax.set_xlim([0,max(ks)])
plt.xticks([0, k_max], ["R", str(lambda_max)+"RX"])
ax.set_xlabel("$\\mathbf{k}$", fontsize=16)
ax.set_ylabel("$1/\\tau_{\\mathbf{k}}$ (THz)", fontsize=16)
ax.set_title("Inverse lifetime in CsPbI$_3$", fontsize=18)
ax.legend(handles=plot_handles, loc="upper right", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=13)
fig.tight_layout()
fig.savefig(plots_folder+"inverse_lifetimes.pdf")

## === Plot of the mobility ===
## ============================

def mobility(temperature):
    """Electron mobility using only linear interaction"""

    # Use analytical SERTA formula for the integrand:
    func = lambda x: 1e4 * e/band_mass*4/(3*np.sqrt(np.pi)) * \
          x**1.5 * np.exp(-x) / tau_inv(x, temperature)
    # Integrate from 0 to infinity:
    return intgr.quad(func, 0, +np.inf)[0]

def mobility2(temperature, directory, params_string):
    """Electron mobility using linear and nonlinear interaction"""

    # Use analytical SERTA formula with full inverse lifetime:
    func = lambda x: 1e4 * e/band_mass*4/(3*np.sqrt(np.pi)) * \
          x**1.5 * np.exp(-x)/(tau_inv(x, temperature) + tau2_inv(x, temperature, directory, params_string))
    # Integrate from 0 to infinity:
    return intgr.quad(func, 0, +np.inf)[0]

mobility_vec = np.vectorize(mobility)

if recalculate:  # Recalculating this should take about 10 minutes
    cutoffs = np.array([0.05,0.1,0.2])
    for cutoff in cutoffs:
        cutoff_str = "_cutoff"+str(cutoff).replace(".","")
        results_directory="results/CsPbI3/super222"+cutoff_str+"/"
        Ts = np.linspace(10., 500., 491)
        Ts2 = np.arange(10, 510, 10)
        mus = mobility_vec(Ts)
        mus2 = np.array([mobility2(T, results_directory, params_string) for T in Ts2])
        save_name = results_directory+"mobility_info.npz"
        create_path(save_name)
        np.savez(save_name, Ts=Ts, Ts2=Ts2, mus=mus, mus2=mus2)


data005 = np.load("results/CsPbI3/super222_cutoff005/mobility_info.npz")
data01 = np.load("results/CsPbI3/super222_cutoff01/mobility_info.npz")
data02 = np.load("results/CsPbI3/super222_cutoff02/mobility_info.npz")
Ts = data01["Ts"]
mus = data01["mus"]
Ts005 = data005["Ts2"]
mus005 = data005["mus2"]
Ts01 = data01["Ts2"]
mus01 = data01["mus2"]
Ts02 = data02["Ts2"]
mus02 = data02["mus2"]

## Power law fit
Tfit_min = 200
Tfit_max = 500
indices0 = (Ts>=Tfit_min)*(Ts<=Tfit_max)
result0 = scipy.stats.linregress(np.log(Ts[indices0]), np.log(mus[indices0]))
indices005 = (Ts005>=Tfit_min)*(Ts005<=Tfit_max)
result005 = scipy.stats.linregress(np.log(Ts005[indices005]), np.log(mus005[indices005]))
indices01 = (Ts01>=Tfit_min)*(Ts01<=Tfit_max)
result01 = scipy.stats.linregress(np.log(Ts01[indices01]), np.log(mus01[indices01]))
indices02= (Ts02>=Tfit_min)*(Ts02<=Tfit_max)
result02 = scipy.stats.linregress(np.log(Ts02[indices02]), np.log(mus02[indices02]))

mus0_fit = np.exp(result0.slope*np.log(Ts)+result0.intercept)
mus01_fit = np.exp(result01.slope*np.log(Ts01)+result01.intercept)
power0_string = "{:.2f}".format(result0.slope)
power01_string = "{:.2f}".format(result01.slope)

# Plot 1: Log-log plot between 100K and 500K, saved 3 times to build up during presentations
# Two basic plots, as well as error range
fig, ax = plt.subplots()
plot_handle1, = ax.plot(Ts, mus, color="#8888FF", linewidth=3.0, label="1e1ph")
plot_handle2, = ax.plot(Ts01, mus01, color="#FF8888", linewidth=3.0, label="1e1ph+1e2ph")
plot_handle3 = ax.fill_between(Ts005, mus005, mus02, color="#FF5656", alpha=0.3, label="Cutoff 0.05-0.2 THz")
plot_handles = [plot_handle1, plot_handle2, plot_handle3]
ax.set_ylim([20,150])
ax.set_xlim([100,500])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Temperature (K)", fontsize=16)
ax.set_ylabel("Mobility (cm²/Vs)", fontsize=16)
ax.set_title("Electron mobility in CsPbI$_3$", fontsize=18)
ax.legend(handles=plot_handles, loc="upper right", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_xticks([100, 200, 300, 400, 500])
ax.set_yticks([20, 30, 40, 60, 100, 150])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
fig.tight_layout()
fig.savefig(plots_folder+"mobility_loglog_1.pdf")
# Relative change inset plot
ax2 = ax.inset_axes([0.1, 0.13, 0.3, 0.3])
mus0 = mobility_vec(Ts01)
plot_handle21, = ax2.plot(Ts01, 1-mus01/mus0, color="#FF8888", linewidth=3)
plot_handle22 = ax2.fill_between(Ts005, 1-mus005/mus0, 1-mus02/mus0, color="#FF5656", alpha=0.3)
plot_handles = [plot_handle21, plot_handle22]
ax2.set_ylim([0,0.2])
ax2.set_xlim([100,500])
ax2.set_xticks([100, 200, 300, 400, 500])
ax2.set_xlabel("Temperature (K)", fontsize=13)
ax2.set_title("Relative change", fontsize=15)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1,0))
fig.savefig(plots_folder+"mobility_loglog_2.pdf")
# Add power law fits
plot_handle4, = ax.plot(Ts, mus0_fit, color=[0.0,0.0,1.0], linestyle="dashed", linewidth=2.0, label="$\\mu \\sim T^{"+power0_string+"}$")
plot_handle5, = ax.plot(Ts01, mus01_fit, color=[1.0,0.0,0.0], linestyle="dashed", linewidth=2.0, label="$\\mu \\sim T^{"+power01_string+"}$")
plot_handles = [plot_handle1, plot_handle2, plot_handle3, plot_handle4, plot_handle5]
ax.legend(handles=plot_handles, loc="upper right", fontsize=14)
fig.savefig(plots_folder+"mobility_loglog_3.pdf")

# Plot 2: Inverse mobility
fig, ax = plt.subplots()
plot_handle1, = ax.plot(Ts, 1/mus, color="blue", label="1e1ph")
plot_handle2, = ax.plot(Ts01, 1/mus01, color="red", label="1e1ph+1e2ph")
plot_handle3 = ax.fill_between(Ts005, 1/mus005, 1/mus02, color="red", alpha=0.3, label="Cutoff 0.05-0.2 THz")
plot_handles = [plot_handle1, plot_handle2, plot_handle3]
ax.set_ylim([0,0.05])
ax.set_xlim([0,max(Ts)])
ax.set_xlabel("Temperature (K)", fontsize=16)
ax.set_ylabel("Inverse mobility (V*s/cm²)", fontsize=16)
ax.set_title("Inverse electron mobility in CsPbI$_3$", fontsize=18)
ax.legend(handles=plot_handles, loc="upper left", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=13)
fig.tight_layout()
fig.savefig(plots_folder+"mobility_inverse.pdf")

### SUPPLEMENTARY INFORMATION PLOTS

## === Noise metrics with respect to ENCUT and dx ===
## ==================================================
Efield_plus = 0.0025
data_folder = "data/CsPbI3"
material_name = "CsPbI3"
supercell_string = "super222"

plot_variables_1 = {
    "parameter_names": ["a6.276_EDIFF3e-06_disp0.005", "a6.276_EDIFF1e-06_disp0.005", "a6.276_EDIFF3e-07_disp0.005",
                        "a6.276_EDIFF1e-07_disp0.005", "a6.276_EDIFF3e-08_disp0.005", "a6.276_EDIFF1e-08_disp0.005",
                        "a6.276_EDIFF3e-09_disp0.005", "a6.276_EDIFF1e-09_disp0.005"],
    "x_values": np.array([3e-06, 1e-06, 3e-07, 1e-07, 3e-08, 1e-08, 3e-09, 1e-09]),
    "x_name": "Electronic loop tolerance",
    "x_unit": "eV",
    "y_lim": [1e-5, 1e2],
    "title": "Electronic loop convergence",
    "save_name": plots_folder+"noise_metrics_EDIFF.pdf",
    "legend_location": "upper left"
}
plot_variables_2 = {
    "parameter_names": ["a6.276_EDIFF1e-08_disp0.01", "a6.276_EDIFF1e-08_disp0.005", "a6.276_EDIFF1e-08_disp0.003",
                        "a6.276_EDIFF1e-08_disp0.002", "a6.276_EDIFF1e-08_disp0.001"],
    "x_values": np.array([0.01, 0.005, 0.003, 0.002, 0.001]),
    "x_name": "Finite displacement",
    "x_unit": "$\\text{\\AA}$",
    "y_lim": [1e-2, 1e1],
    "title": "Finite displacement convergence",
    "save_name": plots_folder+"noise_metrics_dx.pdf",
    "legend_location": "upper right"
}

for plot_index, plot_variables in enumerate([plot_variables_1, plot_variables_2]):
    parameter_names = plot_variables["parameter_names"]
    xvalues = plot_variables["x_values"]
    xname = plot_variables["x_name"]
    xunit = plot_variables["x_unit"]
    ylim = plot_variables["y_lim"]
    title = plot_variables["title"]
    save_name = plot_variables["save_name"]
    legend_location = plot_variables["legend_location"]

    real_normalities = np.empty_like(xvalues)
    imag_normalities = np.empty_like(xvalues)
    der_normalities = np.empty_like(xvalues)
    for index, params_string in enumerate(parameter_names):
        Eplus_file = data_folder+"/"+material_name+"_"+params_string+"_"+"E"+str(Efield_plus)+"z_"+supercell_string+".yaml"
        Ezero_file = data_folder+"/"+material_name+"_"+params_string+"_"+"E0.0z_"+supercell_string+".yaml"
        Eminus_file = data_folder+"/"+material_name+"_"+params_string+"_"+"E"+str(-Efield_plus)+"z_"+supercell_string+".yaml"
        born_file = data_folder+"/BORN_"+material_name+"_"+params_string+".txt"
        calc_plus = PhonopyCommensurateCalculation(Eplus_file, born_filename=born_file)
        calc_zero = PhonopyCommensurateCalculation(Ezero_file, born_filename=born_file)
        calc_minus = PhonopyCommensurateCalculation(Eminus_file, born_filename=born_file)
        qpoints = calc_zero.qpoints[1:]  # Exclude the Gamma point
        dynmat_plus = calc_plus.get_dynamical_matrices(unit="THz", convention="c-type")[1:]
        dynmat_zero = calc_zero.get_dynamical_matrices(unit="THz", convention="c-type")[1:]
        dynmat_minus = calc_minus.get_dynamical_matrices(unit="THz", convention="c-type")[1:]
        dynmat_der = (dynmat_plus-dynmat_minus)/(2*Efield_plus)
        
        real_normality_matrix = np.real(dynmat_plus-dynmat_minus)/np.max(np.abs(np.real(dynmat_plus+dynmat_minus)))
        imag_normality_matrix = np.imag(dynmat_plus+dynmat_minus)/np.max(np.abs(np.imag(dynmat_plus-dynmat_minus)))
        der_normality_matrix = np.real(dynmat_der)/np.max(np.abs(dynmat_der))
        
        real_normalities[index] = np.sum(real_normality_matrix**2)
        imag_normalities[index] = np.sum(imag_normality_matrix**2)
        der_normalities[index] = np.sum(der_normality_matrix**2)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_handle1, = ax.plot(xvalues, imag_normalities, label="Imaginary", color=color_list[0], marker="o")
    plot_handle2, = ax.plot(xvalues, der_normalities, label="Derivative", color=color_list[1], marker="o", linestyle="dashed")
    plot_handles = (plot_handle1, plot_handle2)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xname + " ("+xunit+")", fontsize=13)
    ax.set_ylabel("Noise metric $\\chi^2$", fontsize=13)
    ax.set_xlim([np.min(xvalues), np.max(xvalues)])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=13)
    if plot_index == 1:
        ax.set_xticks([0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])
        ax.set_xticklabels(["0.01", "", "", "", "", "0.005", "", "0.003", "0.002", "0.001"])
    ax.legend(handles=plot_handles, loc=legend_location, fontsize=14)
    fig.savefig(save_name)

## === T(omega) convergence with respect to ENCUT and dx ===
## =========================================================
data_folder = "data/CsPbI3"
text_sizes=(13, 15, 16)
q_mesh_size = 128
sigma = 0.025
temperature = 300
parameter_names = ["a6.276_EDIFF1e-06_disp0.005", "a6.276_EDIFF1e-07_disp0.005",
                   "a6.276_EDIFF1e-08_disp0.005", "a6.276_EDIFF1e-09_disp0.005"]
legend_entries = ["$E_{\\text{tol}}=10^{-6}$ eV", "$E_{\\text{tol}}=10^{-7}$ eV",
                  "$E_{\\text{tol}}=10^{-8}$ eV", "$E_{\\text{tol}}=10^{-9}$ eV"]
save_name = plots_folder+"Tomega_EDIFF_comparison.pdf"
fig, ax = plt.subplots()
plot_handles = []
omega_max = 0
Tomega_max = 0
for index, (params_string, legend_entry) in enumerate(zip(parameter_names, legend_entries)):
    results_filename = "results/CsPbI3/super222_cutoff01/qmesh"+str(q_mesh_size)+"_sigma"+str(sigma)+"_"+params_string+"_T"+str(temperature)+".npz"
    if recalculate:  # Recalculate data from scratch
        print("Recalculating T(ω) at T="+str(int(temperature))+"K with σ="+str(sigma)
              +"THz and a "+str(q_mesh_size)+"x"+str(q_mesh_size)+"x"+str(q_mesh_size)+" q-mesh")
        start_time = time.time()
        Efields = [-0.0025, 0.0, 0.0025]  # units of V/Å
        yaml_filenames = [data_folder+"/CsPbI3_"+params_string+"_E"+str(Efield)+"z_super222.yaml" for Efield in Efields]
        born_filename = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
        calcY = YCalculation(yaml_filenames, Efields, born_filename, take_imag=True)
        results = calcY.calculate_Tomega(q_mesh_size = q_mesh_size, sigma = sigma, 
                                         include_nac="Gonze", temperature = temperature,
                                         q_split_levels = 2, parallel_jobs = 6, imaginary_cutoff = 0.1)
        results.save_npz(results_filename)
        print("Calculation time: "+str(time.time() - start_time)+"s")
    results = TomegaResults(results_filename)
    omega_max = max(omega_max, results.omega[-1])
    Tomega_max = max(Tomega_max, np.max(results.Tomega))
    plot_handle, = ax.plot(results.omega, results.Tomega, label=legend_entry, color=color_list[index])
    plot_handles.append(plot_handle)

ymin, ymax = round_plot_range(0, Tomega_max, clamp_min=0)
ax.set_title("Convergence of $\\mathcal{T}(\\omega)$, $T=300$K",
             fontsize=text_sizes[2])
ax.set_xlabel("Frequency (THz)", fontsize=text_sizes[1])
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
ax.set_xlim(0, omega_max)
ax.set_ylim(ymin, 0.30)
ax.legend(handles=plot_handles, fontsize=text_sizes[0], loc="upper right")
ax.tick_params(axis='both', labelsize=text_sizes[0])

fig.tight_layout()
fig.savefig(save_name)

## === Yplots including SOC ===
imaginary_cutoff = 0.1
Efield = 0.0025
data_folder = "data/CsPbI3-SOC"
path, path_labels = get_path_and_labels("CUB", True)
supercell_string = "super222"
params_string = "a6.276_EDIFF1e-08_disp0.005"
cutoff_str = "_cutoff"+str(imaginary_cutoff).replace(".","")

born_file = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
common_str = data_folder+"/CsPbI3_"+params_string+"_"
Eminus_file = common_str+"E-"+str(Efield)+"z_"+supercell_string+".yaml"
Ezero_file = common_str+"E0.0z_"+supercell_string+".yaml"
Eplus_file = common_str+"E"+str(Efield)+"z_"+supercell_string+".yaml"
savefig_filename = plots_folder+"Yplots/CsPbI3-SOC_"+supercell_string+cutoff_str+"_"+params_string
title_string = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$ for CsPbI$_3$, with SOC"
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
## ============================


## === Spectral function including SOC ===
data_folder = "data/CsPbI3-SOC"
params_string = "a6.276_EDIFF1e-08_disp0.005"
Ezero_file = data_folder+"/CsPbI3_"+params_string+"_E0.0z_super222.yaml"
q_mesh_size = 128
sigma = 0.025
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
direc = "results/CsPbI3-SOC/super222_cutoff01"
fig, ax = plt.subplots()
plot_handles = []
for index, temperature in enumerate(temperatures):
    results_filename = direc+"/qmesh"+str(q_mesh_size)+"_sigma"+str(sigma)+"_"+params_string+"_T"+str(int(temperature))+".npz"
    if recalculate:  # Recalculate data from scratch
        print("Recalculating T(ω) at T="+str(int(temperature))+"K with σ="+str(sigma)
              +"THz and a "+str(q_mesh_size)+"x"+str(q_mesh_size)+"x"+str(q_mesh_size)+" q-mesh")
        start_time = time.time()
        Efields = [-0.0025, 0.0, 0.0025]  # units of V/Å
        yaml_filenames = [data_folder+"/CsPbI3_"+params_string+"_E"+str(Efield)+"z_super222.yaml" for Efield in Efields]
        born_filename = data_folder+"/BORN_CsPbI3_"+params_string+".txt"
        calcY = YCalculation(yaml_filenames, Efields, born_filename, take_imag=True)
        results = calcY.calculate_Tomega(q_mesh_size = q_mesh_size, sigma = sigma, 
                                         include_nac="Gonze", temperature = temperature,
                                         q_split_levels = 2, parallel_jobs = 6, imaginary_cutoff = 0.1)
        results.save_npz(results_filename)
        print("Calculation time: "+str(time.time() - start_time)+"s")
    calc = TomegaResults(results_filename)
    plot_handle, = ax.plot(calc.omega, calc.Tomega, label="$T="+str(temperature)+"$ K", color = color_list[index])
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
ax.set_title("Electron-phonon spectral function with SOC", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(handles=plot_handles, loc="upper right", fontsize=13)
fig.tight_layout()
fig.savefig(plots_folder+"spectral_function-SOC.pdf")
## =======================================


plt.show()