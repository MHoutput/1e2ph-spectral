from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import scipy

THz_to_meV = 4.135667696
data = np.loadtxt("data/CsPbI3/Ponce_tauinvOmega.txt", dtype=float)
frequencies_meV = data[:,0]  # frequencies in meV
tauinvPonce = data[:,1]*3.0385349213821646  # in THz/meV, after applying
# the conversion factor (comes from Poncé data, correctly gives his fig3B)
frequencies_THz = frequencies_meV/THz_to_meV  # frequencies in THz
tauinvOmegas = tauinvPonce*THz_to_meV  # dimensionless, so correct units

# fig, ax = plt.subplots()
# ax.plot(frequencies_THz, tauinvOmegas)
# ax.set_xlim([0,5])
# ax.set_ylim([0,1600])
# ax.set_xlabel("Frequency (THz)")
# ax.set_ylabel("$\\partial \\tau^{-1} / \\partial \\omega$")
# plt.show()

# Visual peak locations and widths, in THz:
peak1s = np.array([2.3, 4.6, 15.8])/THz_to_meV
widths = np.array([0.5, 0.5, 0.5])/THz_to_meV  # Likely a numerical parameter

# Find peak areas and accurate locations
areas = np.empty_like(peak1s)  # Contributions of each peak to tauinv, in THz
omegaLOs = np.empty_like(peak1s)  # peak locations, aka LO frequencies, in THz
for index, (peak, width) in enumerate(zip(peak1s, widths)):
    indices = np.abs(frequencies_THz - peak) <= width
    x = frequencies_THz[indices]
    y = tauinvOmegas[indices]
    area = scipy.integrate.simpson(y, x)
    moment1 = scipy.integrate.simpson(y*x, x)
    areas[index] = area
    omegaLOs[index] = moment1/area

print("LO phonon frequencies (meV): \n"+str(omegaLOs*THz_to_meV))
print("Peak areas (THz): \n"+str(areas))

## Now we can find the mode polarities

# Material parameters:
e = 1.602176634e-19  # Elementary charge, in Coulomb
hbar = 6.62607015e-34/(2*np.pi)  #Reduced Planck constant, in J.s
epsvac = 8.8541878188e-12  # Vacuum permittivity, in C^2/(J.m)
mel = 9.1093837139e-31  # Electron mass, in kg
amu = 1.66053906892e-27  # Atomic mass unit, in kg
kB = 1.380649e-23  # Boltzmann constant in J/K
band_mass = 0.17*mel  # Electron band mass, in kg
lattice_constant = 6.276  # CsPbI3 lattice constant, in Å
epsinf = 6.3  # High-frequency dielectric constant
unit_cell_volume_SI = lattice_constant**3*1e-30  # Unit cell volume, in m^3
omega_nus_SI = omegaLOs*THz_to_meV*1e-3*e/hbar  # Model phonon frequencies in rad/s
T = 300  # Temperature, in Kelvin
x = 1.5  # Electron energy, in k_B*T

# Definition of the auxiliary function:
phi = lambda x, x0: np.sqrt(x0/x)*(np.arcsinh(np.sqrt(x/x0))/(np.exp(x0)-1)
                                   + np.arccosh(np.maximum(1,np.sqrt(x/x0))) \
                                    *(x>x0)/(1-np.exp(-x0)))

for omega, area in zip(omega_nus_SI, areas):
    G2_nu_SI = area*1e12/((2/hbar**2)*unit_cell_volume_SI/(4*np.pi)*np.sqrt(2*band_mass/hbar)) *np.sqrt(omega)/phi(x, hbar*omega/(kB*T))
    p2_nu = G2_nu_SI/((e**2/(epsvac*epsinf*unit_cell_volume_SI))**2*hbar/(2*omega) / 1.66053906892e-27)
    print(p2_nu)


files = ["data/CsPbI3/Ponce_epw1K0meV.txt", 
         "data/CsPbI3/Ponce_epw150K0meV.txt", 
         "data/CsPbI3/Ponce_epw300K0meV.txt"]
fig, ax = plt.subplots()
for file in files:
    data = np.loadtxt(file, dtype=float)
    scattering_rates = data[:, 7]*3.0385349213821646
    ax.plot(scattering_rates, marker="o")
ax.set_xlim([0,20])
ax.set_ylim([0,200])
plt.show()
