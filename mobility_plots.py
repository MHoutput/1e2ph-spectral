import os
import numpy as np
import scipy
import scipy.integrate as intgr
import matplotlib as mpl
import matplotlib.pyplot as plt
from phonopyReaders import PhonopyCommensurateCalculation, YCalculation, \
    TomegaResults, round_plot_range

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble']="\\usepackage{bm}"
mpl.rcParams['text.latex.preamble']="\\usepackage{amsmath}"
np.set_printoptions(precision=6, suppress=True)

## Plot of T(omega)
temperatures = [0,100,200,300]
omega_nus = np.array([0.402821, 0.964438, 3.904019])  # Model phonon frequencies, in THz
p2_nus = np.array([0.002726, 0.005809, 0.307629])  # Squared mode polarities, in (a.m.u.)^-1
delta_areas = (1.602176634e-19)**2/(2*8.8541878188e-12*(6.24e-10)**3)*(p2_nus/1.66053906892e-27)/(omega_nus*(2*np.pi*1e12)**2)
annotate_ys = np.array([0.23, 0.21, 0.1])
arrowprops=dict(arrowstyle="->, head_width=0.1", relpos=(0., 0.5))
direc = "choose_results/CsPbI3/precise/super222/"
fig, ax = plt.subplots()
plot_handles = []
for temperature in temperatures:
    calc = TomegaResults(direc+"qmesh256_sigma0.01_a6.24_T"+str(int(temperature))+".npz")
    plot_handle, = ax.plot(calc.omega, calc.Tomega, label="$T="+str(temperature)+"$ K")
    plot_handles.append(plot_handle)
ax.vlines(omega_nus, 0, 0.25, color="black", linewidth=0.7)
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
ax.set_ylim([0,0.25])
ax.set_xlabel("Frequency (THz)", fontsize=16)
ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=16)
ax.set_title("1-electron-2-phonon spectral function", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(handles=plot_handles, loc="upper right", fontsize=13)
fig.tight_layout()
fig.savefig("mobility_plots/SpectralFunction.pdf")
plt.show()