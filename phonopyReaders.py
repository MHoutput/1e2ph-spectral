"""
Classes that read and handle PhonoPy output .yaml files

Exported classes
----------------
PhonopyCalculation: base class
PhonopyMeshCalculation: read mesh calculation and calculate DOS
PhonopyBandCalculation: read band calculation along a path
PhonopyCommensurateCalculation: read and process commensurate points data
YCalculation: read finite E-field calculations and calculate T(omega)

Exported functions
------------------
get_modular_indices: get digits of number in a varying base
round_plot_range: get ronded axis limits for plotting
create_path: create a path if it doesn't already exist
n_BE: Bose-Einstein distribution function

Written by Matthew Houtput (matthew.houtput@uantwerpen.be)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import scipy
import yaml
import warnings
import joblib
import itertools

def get_modular_indices(number, mod_list):
    """ Breaks the input number up into a list of "digits" 
    
    Arguments
    ---------
    number: int, number to be written in modular indices
    mod_list: list of int, moduli used for digits
    
    Returns
    -------
    modular_indices: list of int, the digits for the number
    
    Most commonly used to traverse or create arrays with shape mod_list.
    If mod_list = [b, b, b, ...], the digits correspond to the digits
      of the input number in base b.
    """

    mod = np.prod(mod_list)
    if len(mod_list) == 0:
        return np.array([number])
    if len(mod_list) == 1:
        return np.array([number % mod, number // mod])
    else:
        new_list = mod_list[0:-1]
        return np.append(get_modular_indices(number % mod, new_list), 
                         number // mod)
    
def round_plot_range(ymin, ymax, clamp_min=None, clamp_max=None, targets=None):
    """ Returns rounded plot limits based on min and max of data
    
    Arguments
    ---------
    ymin: minimum y-value of the data on the plot
    ymax: maximum y-value of the data on the plot
    clamp_min: fixed lower limit, default None
    clamp_max: fixed upper limit, default None
    targets: list of real, round numbers used as rounding targets
        Default:    [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 
                    3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    Returns
    -------
    ymin_rounded: lower limit for the plot, equals clamp_min if not None
    ymax_rounded: upper limit for the plot, equals clamp_max if not None
    
    """
    
    if targets is None:
        targets = [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 
                   3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ceil_to = lambda x: targets[np.nonzero(targets > x)[0][0]]

    if clamp_min is not None:
        ymin_rounded = clamp_min
        if clamp_max is not None:
            ymax_rounded = clamp_max
        else:
            # ymax > clamp_min
            rounding_scale = 10**np.floor(np.log10(ymax-clamp_min))
            ymax_rounded = clamp_min+ceil_to((ymax-clamp_min)/rounding_scale)\
                *rounding_scale
    else:
        if clamp_max is not None:
            # ymin < clamp_max
            ymax_rounded = clamp_max
            rounding_scale = 10**np.floor(np.log10(clamp_max-ymin))
            ymin_rounded = clamp_max-ceil_to((clamp_max-ymin)/rounding_scale)\
                *rounding_scale
        else:
            # ymax > ymin
            scale_avg = 10**np.floor(np.log10(ymax-ymin))
            avg = 0.5*(ymin+ymax)
            avg_round = round(avg/scale_avg)*scale_avg
            scale_min = 10**np.floor(np.log10(avg_round-ymin))
            ymin_rounded = avg_round-ceil_to((avg_round-ymin)/scale_min)\
                *scale_min
            scale_max = 10**np.floor(np.log10(ymax-avg_round))
            ymax_rounded = avg_round+ceil_to((ymax-avg_round)/scale_max)\
                *scale_max
    return ymin_rounded, ymax_rounded

def create_path(filename):
    """ Create necessary directories to save a file 

    When trying to save a file with a filename that contains
    directories, the save fails if those directories do not exist yet.
    This function creates the necessary directories that are present 
    in the name of the file, if they do not exist yet.
    
    Arguments
    ---------
    filename: str
        Name of the file to be saved, or path to be created

    """
    path_name, _ = os.path.split(filename)
    path_norm = os.path.normpath(path_name)
    spl_char = os.path.sep
    dirs_to_check = [spl_char.join(path_norm.split(spl_char)[:i]) 
                     for i in range(1, len(path_norm.split(spl_char))+1)]
    for direc in dirs_to_check:
        if not os.path.isdir(direc):
            os.mkdir(direc)

def n_BE(omega, temp):
    """ Bose-Einstein distribution for phonon frequencies

    Arguments
    ---------
    omega: np.array of real
        Phonon cycle frequencies at which to evaluate n_BE(omega),
        in units of THz
    temp: real
        Temperature at which to evaluate n_BE

    Returns
    -------
    n_BE: np.array of real, same shape as omega
        Bose-Einstein distribution of the given omega

    Raises
    ------
    ValueError
        when temp is smaller than zero

    """
    if temp < 0:
        raise ValueError("negative temperatures are not allowed")
    if temp < 1e-10:
        return (omega >= 0)-1
    else:
        return 1/(np.exp(47.9924307*omega/temp)-1)


class PhonopyCalculation:
    """ Base class that all other classes inherit from

    This class cannot be instanciated. Create instances of the
    following child classes instead:
    - PhonopyMeshCalculation
    - PhonopyBandCalculation
    - PhonopyCommensurateCalculation
    - YCalculation
    
    Attributes
    ----------
    supercell_size: np.array of int
        shape (3,)
    reciprocal_lattice_vectors: np.array of real
        shape (3,3), units of inverse Angstroms
    lattice_vectors: np.array of real
        shape (3,3), units of Angstroms
    unitcell_volume: real
        units of cubic Angstroms
    num_dimensions: int
        Equal to 3
    natom: int
    numqpoints: int
    labels: list of (list of str) if band calculation, else None
    segment_nqpoint: list of int if band calculation, else None
    numbands: int
        Equal to 3*natom
    qpoints: np.array of real
        shape (numqpoints, 3), in direct coordinates
    distances: np.array of real
        shape (numqpoints,)
    weights: np.array of int
        shape (numqpoints,)
    frequencies: np.array of real
        shape (numqpoints, numbands), stored as cycle frequencies in THz
    eigenvectors: np.array of complex, 
        shape (numqpoints, numbands, natom, 3)
    atom_names: np.array of str
        shape (natom,)
    atom_masses: np.array of real
        shape (natom,), in atomic mass units
    atom_positions: np.array of real
        shape (natom, 3), in direct coordinates
    born_is_set: bool
        True if born_filename is read
    born_charges: np.array of real
        shape (natom, 3, 3)
    dielectric_tensor: np.array of real
        shape (3,3)

    Raises
    ------
    TypeError
        when trying to create an instance of this class

    """
    
    def __new__(cls, *args, **kwargs):
        """ Ensure this class cannot be instanciated """
        if cls is PhonopyCalculation:
            raise TypeError(f"""only children of '{cls.__name__}' 
                             may be instantiated""")
        return object.__new__(cls)
    
    def __init__(self, yaml_filename, born_filename=None):
        """ PhonopyCalculation(yaml_filename, born_filename)

        Arguments
        ---------
        yaml_filename: string
            .yaml file exported by PhonoPy
        born_filename: string
            BORN file that contains the Born effective charge tensors
            and dielectric tensors.
            Important: this function expects a BORN file with a line
            for each atom, since no symmetry is implemented. PhonoPy
            exports a BORN file with less lines based on symmetry,
            which is not compatible with this code.
        """
        self.load_yaml(yaml_filename)
        self.load_BORN(born_filename)
                    
    def load_yaml(self, yaml_filename):
        """ Load data from PhonoPy .yaml file

        Automatically called from the constructor. Reads all data from
        the PhonoPy .yaml file and stores them in the relevant class
        attributes.

        Arguments
        ---------
        yaml_filename: string
            .yaml file exported by PhonoPy to load in
        """
        with open(yaml_filename, 'r') as file:
            yaml_string = file.read()
            file.close()
            
        mesh_dict = yaml.safe_load(yaml_string)
        self.supercell_size = np.array(mesh_dict.get('mesh', None))
        self.reciprocal_lattice_vectors = \
            np.array(mesh_dict['reciprocal_lattice'])
        self.lattice_vectors = np.array(mesh_dict['lattice'])
        self.unitcell_volume = np.abs(np.linalg.det(self.lattice_vectors))
        self.num_dimensions = len(self.lattice_vectors[0])
        self.natom = mesh_dict['natom']
        self.numqpoints = mesh_dict['nqpoint']
        self.labels = mesh_dict.get('labels', None)
        self.segment_nqpoint = mesh_dict.get('segment_nqpoint', None)
        phonon_props = mesh_dict['phonon']
        eigenvectors_exist = 'eigenvector' in phonon_props[0]['band'][0]
        self.numbands = self.num_dimensions*self.natom
        self.qpoints = np.empty((self.numqpoints,self.num_dimensions))
        self.distances = np.empty((self.numqpoints,))
        self.weights = np.empty((self.numqpoints,))
        self.frequencies = np.empty((self.numqpoints, self.numbands))
        if eigenvectors_exist:
            self.eigenvectors = np.empty((self.numqpoints, self.numbands, 
                                          self.natom, self.num_dimensions), 
                                         dtype=np.complex_)
        else:
            self.eigenvectors = None
        for index, point in enumerate(phonon_props):
            self.qpoints[index] = point['q-position']
            if 'distance_from_gamma' in point:
                self.distances[index] = point['distance_from_gamma']
            if 'distance' in point:
                self.distances[index] = point['distance']
            self.weights[index] = point.get('weight', 1)
            bands = point['band']
            for nu, band in enumerate(bands):
                self.frequencies[index, nu] = band['frequency']
                if eigenvectors_exist:
                    eigenvector_list = np.array(band['eigenvector'])
                    self.eigenvectors[index, nu] = \
                        eigenvector_list[:,:,0]+eigenvector_list[:,:,1]*1j
        atom_info = mesh_dict.get('points')
        self.atom_names = np.empty((self.natom,), dtype=np.dtypes.StrDType)
        self.atom_masses = np.empty((self.natom,))
        self.atom_positions = np.empty((self.natom, self.num_dimensions))
        for index, atom in enumerate(atom_info):
            self.atom_names[index] = atom['symbol']
            self.atom_masses[index] = atom['mass']
            self.atom_positions[index] = atom['coordinates']
            
    def load_BORN(self, born_filename):
        """ Load data from BORN file

        Automatically called from the constructor. Reads all data from
        the PhonoPy BORN file and stores them in the relevant class
        attributes.
        Important: this function expects a BORN file with a line
        for each atom, since no symmetry is implemented. PhonoPy
        exports a BORN file with less lines based on symmetry,
        which is not compatible with this code.

        Arguments
        ---------
        born_filename: string
            BORN file to load in
        """
        if born_filename is None:
            self.born_is_set = False
            self.born_charges = None
            self.dielectric_tensor = None
        else:
            self.born_is_set = True
            loaded_array = np.loadtxt(born_filename, dtype=float, comments="#")
            self.dielectric_tensor = loaded_array[0,...].reshape((3,3))
            self.born_charges = loaded_array[1:,...].reshape((-1,3,3))
    
    def get_supercell_size(self):
        """ Return the supercell size """
        return self.supercell_size
    
    def get_unitcell_volume(self):
        """ Return the unit cell volume in cubic Angstroms """
        return self.unitcell_volume
    
    def get_distances(self):
        """ Return the distances along a q-point path """
        return self.distances
    
    def get_eigenvectors(self, convention="c-type"):
        """ Return the phonon eigenvectors as a 4D array

        Can return either the eigenvectors in the c-type convention
        (default convention used by PhonoPy), or the d-type convention.

        Arguments
        ---------
        convention: string
            Must be either "c-type" or "d-type"

        Returns
        -------
        eigenvectors: np.array of complex
            shape (numqpoints, numbands, natom, num_dimensions)

        Raises
        ------
        NameError
            when input convention is neither "c-type" nor "d-type"
        """
        match convention:
            case "c-type":
                return self.eigenvectors
            case "d-type":
                phases = np.exp(2*np.pi*1j* self.get_qpoints() 
                                @ self.get_atom_positions().T)
                return ((phases.T) * (self.eigenvectors.transpose((3,1,2,0))))\
                    .transpose(3,1,2,0)
            case _:
                raise NameError("convention must be c-type or d-type")
    
    def get_eigenvectors_matrices(self, convention="c-type"):
        """ Return the phonon eigenvectors as a 3D array

        Can return either the eigenvectors in the c-type convention
        (default convention used by PhonoPy), or the d-type convention.

        Arguments
        ---------
        convention: string
            Must be either "c-type" or "d-type"

        Returns
        -------
        eigenvectors: np.array of complex
            shape (numqpoints, numbands, numbands)

        Raises
        ------
        NameError
            when input convention is neither "c-type" nor "d-type"
        """
        return self.get_eigenvectors(convention)\
            .reshape(-1,self.numbands,self.numbands)
    
    def get_frequencies(self, unit="THz"):
        """ Return the phonon frequencies in the desired units

        Arguments
        ---------
        unit: string
            Must be one of the units supported in convert_units

        Returns
        -------
        frequencies: np.array of real
            shape (numqpoints, numbands)
            Imaginary frequencies are output as negative frequencies

        Raises
        ------
        NameError
            when input unit is not one of the recognized units
        """
        return self.convert_units(self.frequencies, to_unit=unit)
    
    def get_frequencies_matrices(self, unit="THz"):
        """ Return phonon frequencies as a stack of diagonal matrices

        Arguments
        ---------
        unit: string
            Must be one of the units supported in convert_units

        Returns
        -------
        frequencies: np.array of real
            shape (numqpoints, numbands, numbands)
            Imaginary frequencies are output as negative frequencies

        Raises
        ------
        NameError
            when input unit is not one of the recognized units
        """
        frequencies = self.get_frequencies(unit)
        frequencies_matrices = np.empty((self.numqpoints, self.numbands, 
                                         self.numbands))
        for index, freqs in enumerate(frequencies):
            frequencies_matrices[index,...] = np.diag(freqs)
        return frequencies_matrices
    
    def get_dynamical_matrices(self, unit="THz", convention="c-type"):
        """ Calculate and return dynamical matrices

        Arguments
        ---------
        unit: string
            Must be one of the units supported in convert_units

        convention: string
            Must be either "c-type" or "d-type"

        Returns
        -------
        dynamical_matrices: np.array of complex
            shape (numqpoints, numbands, numbands)
            Dynamical matrices evaluated at self.qpoints

        Raises
        ------
        NameError
            when input unit is not one of the recognized units
        """
        frequencies_matrices = self.get_frequencies_matrices(unit)
        eigenvectors_matrices = self.get_eigenvectors_matrices(convention)
        dynamical_matrices = np.empty((self.numqpoints, self.numbands, 
                                       self.numbands), dtype=np.complex_)
        for index, freqs_vecs in enumerate(zip(frequencies_matrices, 
                                               eigenvectors_matrices)):
            freqs_squared = np.sign(freqs_vecs[0])*np.power(freqs_vecs[0], 2)
            eigvecs = freqs_vecs[1]
            dynamical_matrices[index,...] = \
                eigvecs.T @ freqs_squared @ eigvecs.conj()
        return dynamical_matrices        
    
    def get_lattice_vectors(self):
        """ Return the lattice vectors in Angstroms """
        return self.lattice_vectors

    def get_reciprocal_lattice_vectors(self):
        """ Return the reciprocal lattice vectors in inverse Angstroms """
        return self.reciprocal_lattice_vectors
    
    def get_qpoints(self):
        """ Return the list of q-points included in the dataset """
        return self.qpoints
    
    def get_lpoints(self):
        """ Return a list of lattice vectors in the supercell """
        modulos = self.supercell_size
        return np.array([get_modular_indices(n, modulos[0:-1]) 
                         for n in range(np.prod(modulos))])
    
    def get_dense_qmesh(self, size, fixed_indices=np.array([]), 
                        num_dimensions=None):
        """ Return a mesh of q-points for Brillouin zone integration

        Arguments
        ---------
        size: int or np.ndarray() of int, shape (3,)
            Number of q-points in each direction (e.g. 16x16x16)
            If int, uses the same number for all directions
            If np.ndarray(), uses these numbers for each direction

        fixed_indices: np.ndarray() of real
            Keep one or more coordinates fixed. For example, if
            fixed_indices = np.array([0.1, 0.2]), the coordinates will
            be of the form [q1, 0.1, 0.2] with varying q1
            Useful for very large meshes where it is impractical to
            store the entire mesh in one array
            Default: No indices fixed

        num_dimensions: int
            Number of coordinates, usually 3
            Default: self.num_dimensions

        Returns
        -------
        q_mesh: np.array of real
            shape (size**3, 3) or (prod(size), 3)
            Array of q-points in the mesh, in direct coordinates

        Raises
        ------
        ValueError
            when size is not an int or np.ndarray()
        
        """
        if num_dimensions is None:
            num_dimensions=self.num_dimensions
        num_free = num_dimensions - len(fixed_indices)
        match size:
            case int():
                modulos = np.repeat(size, num_free)
            case np.ndarray():
                modulos = np.floor(size).astype(int).flatten()[0:num_free]
            case _:
                raise ValueError("""size must be one integer 
                                 or a numpy array of integers""")
        total_size = np.prod(modulos)
        result = np.empty((total_size,num_dimensions))
        result[:,:num_free] = np.array([get_modular_indices(n, modulos[0:-1]) 
                                        for n in range(total_size)]) / modulos
        result[:,num_free:] = fixed_indices
        return result
    
    def get_Gpoints(self, cutoff_radius, include_zero=True):
        """ Generate list of reciprocal lattice points around Gamma
        
        Arguments
        ---------
        cutoff_radius: real
            All reciprocal lattice points within a reciprocal distance
            cutoff_radius of Gamma are included in the list

        include_zero: bool
            Output does not include G=0 if set to false

        Returns
        -------
        Gpoints: np.array of real
            shape (:, 3), array of reciprocal lattice points

        """
        metric = self.reciprocal_lattice_vectors @ \
            self.reciprocal_lattice_vectors.T
        Gnorm = lambda G: np.linalg.norm(G @ self.reciprocal_lattice_vectors, 
                                         axis=-1)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        m_cutoff = np.ceil(cutoff_radius \
                           /np.sqrt(np.min(eigenvalues))).astype(int)
        modulos = (2*m_cutoff+1)*np.ones((self.num_dimensions-1,), dtype=int)
        Gpoints = np.array([get_modular_indices(n, modulos) - m_cutoff 
                            for n in range((2*m_cutoff+1)**self.num_dimensions) 
                            if Gnorm(get_modular_indices(n, modulos) - m_cutoff)
                              < cutoff_radius])
        if not include_zero:
            for zero_index in np.nonzero(np.linalg.norm(Gpoints, ord=2, axis=1) 
                                         < 1e-10)[0]:
                Gpoints = np.delete(Gpoints, zero_index, axis=0)
        return Gpoints  
    
    def get_weights(self):
        """Return weights for Brillouin zone integration"""
        return self.weights
    
    def get_atom_names(self):
        """Return names of atoms in the unit cell"""
        return self.atom_names
    
    def get_atom_masses(self):
        """Return masses of atoms in the unit cell"""
        return self.atom_masses
    
    def get_mass_matrix(self):
        """Return masses of atoms in diagonal matrix form
        
        Returns
        -------
        maxx_matrix: np.array of real
            shape (self.numbands, self.numbands)
        """
        return np.diag(np.tile(self.get_atom_masses(), 
                               (self.num_dimensions,1)).T.reshape(-1))
    
    def get_atom_positions(self):
        """Return positions of atoms in the unit cell"""
        return self.atom_positions
    
    def get_atom_positions_3N(self):
        """ Returns the atom positions repeated 3 times in a single array
        
        Useful for the dynamical matrix conventions
        """
        return np.reshape(np.array([np.tile(self.get_atom_positions()[i], 
                                            (self.num_dimensions,1)) 
                                    for i in range(self.natom)]),
                          (self.numbands, self.num_dimensions))
    
    def get_tauk_difference(self):
        """ Returns the quantity tau_k - tau_k'
        
        Useful to change between dynamical matrix conventions

        Returns
        -------
        tauk_difference: np.array of real
            shape (self.numbands, self.numbands, 3)
        """
        return ( self.get_atom_positions_3N().reshape(self.numbands, 1, 
                                                      self.num_dimensions)
                -self.get_atom_positions_3N().reshape(1, self.numbands, 
                                                      self.num_dimensions) )
    
    def get_c_to_d_factors(self, qs):
        """ Factors to convert from c-type to d-type convention
        
        Arguments
        ---------
        qs: np.array of real
            shape (:, 3)
            q-points in which the dynamical matrix is to be calculated

        Returns
        -------
        conversion_factors: np.array of complex
            shape (len(qs), self.numbands, self.numbands)
            Conversion factors to convert a stack of dynamical matrices
            in the c-type convention to the d-type convention
        """
        return np.exp(2*np.pi*1j*np.moveaxis(self.get_tauk_difference() @ qs.T, 
                                             [0,1,2], [1,2,0]))
    
    def get_d_to_c_factors(self, qs):
        """ Factors to convert from d-type to c-type convention

        Equal to the complex conjugate of self.get_d_to_c_factors
        
        Arguments
        ---------
        qs: np.array of real
            shape (:, 3)
            q-points in which the dynamical matrix is to be calculated

        Returns
        -------
        conversion_factors: np.array of complex
            shape (len(qs), self.numbands, self.numbands)
            Conversion factors to convert a stack of dynamical matrices
            in the d-type convention to the c-type convention
        """
        return np.exp(-2*np.pi*1j*np.moveaxis(self.get_tauk_difference() @ qs.T,
                                               [0,1,2], [1,2,0]))
    
    def convert_units(self, frequencies, from_unit="THz", to_unit="THz"):
        """ Convert phonon frequencies between units

        The supported units are:
            - "THz": cycle frequencies in THz
            - "rad/s": radial frequencies in rad/s
            - "cm-1": inverse wavelengths in inverse cm
            - "eV": phonon energies in eV
            - "PhonoPy": internal units used by PhonoPy

        Arguments
        ---------
        frequencies: np.array of real
            Frequencies expressed in from_unit
        from_unit: string
        to_unit: string

        Returns
        -------
        frequencies: np.array of real
            Frequencies expressed in to_unit

        Raises
        ------
        warning
            when from_unit or to_unit is not one of the supported units
        """

        match from_unit:
            case "THz":
                frequencies_in_THz = frequencies
            case "rad/s":
                frequencies_in_THz = frequencies/(2*np.pi*1e12)
            case "cm-1":
                frequencies_in_THz = frequencies/33.356409529
            case "eV":
                frequencies_in_THz = frequencies/4.135667696e-3
            case "PhonoPy":
                frequencies_in_THz = frequencies*15.633302
            case _:
                warn_string = str(from_unit)+\
                """ is not a recognized phonon frequency unit.
                Currently only 'THz', 'rad/s', 'cm-1',  'eV', and 'PhonoPy'
                are supported. It is assumed that the input frequencies
                were in THz."""
                warnings.warn(warn_string)
                frequencies_in_THz = frequencies
        
        match to_unit:
            case "THz":
                return frequencies_in_THz
            case "rad/s":
                return frequencies_in_THz*(2*np.pi*1e12)
            case "cm-1":
                return frequencies_in_THz*33.356409529
            case "eV":
                return frequencies_in_THz*4.135667696e-3
            case "PhonoPy":
                return frequencies_in_THz/15.633302
            case _:
                warn_string = str(to_unit)+\
                """ is not a recognized phonon frequency unit.
                Currently only 'THz', 'rad/s', 'cm-1', 'eV', and 'PhonoPy'
                are supported. Frequencies in THz were returned instead."""
                warnings.warn(warn_string)
                return frequencies_in_THz
    
    def get_clean_frequencies(self, unit='THz', cutoff=None, min_value=None, 
                              frequencies_to_clean=None):
        """ Remove any negative frequencies and small frequencies

        Default behavior: Set any negative requencies to 1e-10 THz 
        if their absolute value is smaller than 0.1 THz
        Throws a warning when large negative frequencies are detected

        Arguments
        ---------
        unit: string, frequency unit in which inputs are given
            Default: "THz"
        cutoff: real, threshold for determining small frequencies
            Default: 0.1 THz
        min_value: real, set small frequencies to this value 
            Default: 1e-10 THz
        frequencies_to_clean: np.array of real
            Array of frequencies that must be cleaned in the above way
            Default: self.frequencies

        Returns
        -------
        clean frequencies: np.array of real
            Array of positive frequencies with shape equal to that
            of frequencies_to_clean

        Raises
        ------
        warning
            when one of the frequencies is smaller than -cutoff, which
            indicates a significantly unstable phonon mode

        """

        if frequencies_to_clean is None:
            clean_frequencies = self.get_frequencies(unit=unit)
        else:
            clean_frequencies = 1.0*frequencies_to_clean
        if cutoff is None:
            cutoff = self.convert_units(0.1, from_unit='THz', to_unit=unit)
        if min_value is None:
            min_value = self.convert_units(1e-10, from_unit='THz', to_unit=unit)
        indices = np.nonzero(clean_frequencies < min_value)
        throw_warning = False
        for index, frequency in enumerate(clean_frequencies[indices]):
            clean_frequencies[indices[0][index], indices[1][index]] = min_value
            if frequency < -cutoff:
                throw_warning = True
        if throw_warning:
            warn_string = "Negative frequencies smaller than -"+str(cutoff)+" "\
                          +unit+" detected: material is likely unstable"
            warnings.warn(warn_string)
        return clean_frequencies
    
    def clean_frequencies(self, cutoff=0.1):
        """ Call get_clean_frequencies() on self.clean_frequencies """
        self.frequencies = self.get_clean_frequencies(cutoff=cutoff)
        return self.frequencies
    
    def clean_qpoints(self):
        """ Reduce all q-point coordinates to the range ]-0.5,0.5] """
        normalize_to_range = lambda x: ((x - 0.5) % -1) + 0.5
        self.qpoints = normalize_to_range(self.qpoints)
        return self.qpoints
    
    def parse_path(self, path, path_labels, npoints=51):
        """Convert a given path to a list of q-points and plot inputs

        Arguments
        ---------
        path: list of list of list of real
            High-symmetry path, written in direct coordinates in the same
            conventions as PhonoPy and pathsLabels.py
                - First level: list of connected path segments
                - Second level: list of points that mark path segments
                - Third level: direct coordinates of points
        path_labels: list of str
            List of names of the edge points of the path, in order,
            written in LaTeX markup
        npoints: number of q-points on each segment

        Returns
        -------
        qs: np.array of real
            shape(:, 3), list of q-points on the path
        distances: np.array of real
            shape(:), reciprocal distances along the path, used as
            x-axis data on a plot of phonon bands
        xaxis_labels: list of str
            Size: one more than number of segments in the path
            List of labels of special points, to plot on the x-axis
            on a plot of phonon bands
        jump_indices: np.array of int
            Indices where the path shows a discontinuous jump

        """
        qs = []
        distances_to_nearest = []
        xaxis_labels = [path_labels[0]]
        label_count = 1
        do_between_code = False
        for mini_path in path:
            if do_between_code:
                # Change the last label A to something of the form "A|B"
                # to indicate a discontinuous jump
                xaxis_labels[-1] += "$|$" + path_labels[label_count]
                label_count += 1
            else:
                do_between_code = True
            qs_mini = []
            for i in range(len(mini_path)-1):
                qs_to_append = np.linspace(mini_path[i], mini_path[i+1], 
                                           npoints)
                qs.append(qs_to_append)
                qs_mini.append(qs_to_append)
                xaxis_labels.append(path_labels[label_count])
                label_count += 1
            qs_mini = np.array(qs_mini).reshape((-1,3))
            # Calculate list of distances between two neighbouring points
            qs_cartesian = qs_mini @ self.get_reciprocal_lattice_vectors()
            distances_mini = np.zeros((len(qs_mini),), dtype=float)
            distances_mini[1:] = np.linalg.norm(qs_cartesian[1:]-\
                                                qs_cartesian[:-1], axis=1)
            distances_to_nearest.extend(distances_mini.tolist())
        jump_indices = np.cumsum((np.array([len(x) for x in path])-1)\
                                 *npoints)[:-1] - 1
        qs = np.array(qs).reshape((-1,3))
        distances = np.cumsum(np.array(distances_to_nearest))
        return qs, distances, xaxis_labels, jump_indices
    
    def get_Brillouin_boundary(self, reciprocal_lattice_vectors=None):
        """ Calculates corners, edges, and planes of the Brillouin zone

        Returns the Miller indices of the planes that make up the edge
        of the first Brillouin zone, as well as a list of all the
        corners and edges on its surface. This is useful for plotting
        the Brillouin zone.
        Only works when self.num_dimensions = 3
        This code is heavily based off the code found at
        http://lampz.tugraz.at/~hadley/ss1/bzones/drawing_BZ.php

        Arguments
        ---------
        reciprocal_lattice_vectors: np.array of real
            shape (3,3), units of inverse Angstroms
            Default: self.reciprocal_lattice_vectors
        
        Returns
        -------
        miller_indices: np.array of int
            shape (:,3)
            Miller indices of faces of the Brillouin zone
        corners: np.array of real
            shape (:,3)
            Cartesian coordinates of the corners of the Brillouin zone
        edges: np.array of real
            shape (:,2,3)
            Cartesian coordinates of pairs of corners that define
            the edges of the Brillouin zone
        edge_planes: np.array of int
            shape (:,2,3)
            Miller indices of pairs of planes that intersect at the
            edges of the Brillouin zone

        """
        
        if reciprocal_lattice_vectors is None:
            reciprocal_lattice_vectors = self.get_reciprocal_lattice_vectors()
        num_dimensions = len(reciprocal_lattice_vectors)
        
        # Get the 26 G-vectors surrounding Gamma
        cutoff = 1
        modulos = (2*cutoff+1)*np.ones((num_dimensions-1,), dtype=int)
        Gpoints = np.array([get_modular_indices(n, modulos) - cutoff 
                            for n in range((2*cutoff+1)**num_dimensions)])
        for zero_index in np.nonzero(np.linalg.norm(Gpoints, ord=2, axis=1) 
                                     < 1e-10)[0]:
            Gpoints = np.delete(Gpoints, zero_index, axis=0)
        Gcart = Gpoints @ reciprocal_lattice_vectors
        
        # Find the distances from the planes to Gamma and the other reciprocal 
        # points
        accepted_Gs = []
        accepted_Gscart = []
        for index, G1cart in enumerate(Gcart):
            gamma_distance = np.linalg.norm(0.5*G1cart, ord=2)
            G_distances = np.linalg.norm(0.5*G1cart - Gcart, ord=2, axis=1)
            G_distances[index] = np.nan
            if np.nanmin(G_distances) > gamma_distance:
                accepted_Gs.append(Gpoints[index])
                accepted_Gscart.append(G1cart)
        
        # Find all corners as intersections of three planes
        corners = []
        for G1, G2, G3 in itertools.combinations(accepted_Gscart, 3):  
            # Iterate over all triplets of planes 
            system_matrix = np.array([G1,G2,G3])
            if np.linalg.det(system_matrix) != 0:
                system_RHS = 0.5*np.linalg.norm(system_matrix, ord=2, axis=1)**2
                solution = np.linalg.solve(system_matrix, system_RHS)
                corners.append(solution)
        accepted_corners = []
        for index, corner in enumerate(corners):
            gamma_distance = np.linalg.norm(corner, ord=2)
            G_distances = np.linalg.norm(corner - Gcart, ord=2, axis=1)
            if np.nanmin(G_distances) - gamma_distance >= -1e-10:
                accept_corner = True
                for corner2 in accepted_corners:
                    if np.min(np.linalg.norm(corner-corner2, ord=2)) < 1e-10:
                        accept_corner = False  # Only accept unique corners
                if accept_corner:
                    accepted_corners.append(corner)
                    
        # Find all edges, by checking every pair of corners and whether they 
        # are both on the same two planes
        accepted_edges = []
        accepted_edge_planes = []
        for G1, G2 in itertools.combinations(accepted_Gs, 2):  
            # Iterate over all pairs of planes
            G1_cart = G1 @ reciprocal_lattice_vectors
            G2_cart = G2 @ reciprocal_lattice_vectors
            for corner1, corner2 in itertools.combinations(accepted_corners, 2):  
                # Iterate over all pairs of corners
                distance_11 = np.abs(G1_cart@corner1-0.5*G1_cart@G1_cart)
                distance_12 = np.abs(G2_cart@corner1-0.5*G2_cart@G2_cart)
                distance_21 = np.abs(G1_cart@corner2-0.5*G1_cart@G1_cart)
                distance_22 = np.abs(G2_cart@corner2-0.5*G2_cart@G2_cart)
                if (distance_11 < 1e-10 and distance_12 < 1e-10 \
                    and distance_21 < 1e-10 and distance_22 < 1e-10):
                    # The edge is defined by two corners
                    accepted_edges.append([corner1, corner2]) 
                    # We also keep the two planes that the edges are on
                    accepted_edge_planes.append([G1, G2])  
        
        miller_indices = np.array(accepted_Gs)
        corners = np.array(accepted_corners)
        edges = np.array(accepted_edges)
        edge_planes = np.array(accepted_edge_planes)
        return miller_indices, corners, edges, edge_planes
    
    def plot_Brillouin(self, reciprocal_lattice_vectors=None, path=[], 
                       path_labels=[], label_shifts=None, view_angles=None, 
                       save_filename=None, save_bbox_extents=None, 
                       quiver_plot=None, quiver_labels=None, 
                       visible_linestyle=None, invisible_linestyle=None, 
                       path_linestyle=None, label_style=None, 
                       quiver_style=None, quiver_label_style=None):
        """ Makes a plot of the first Brillouin zone in 3D

        Automatically determines which edges will be visible or
        invisible, given the view angle, and plots the edges in two
        different styles accordingly. Style arguments should be passed
        as dicts. Can save the figure to an external .pdf file

        Arguments
        ---------
        reciprocal_lattice_vectors: np.array of real
            shape (3,3), units of inverse Angstroms
            Default: self.reciprocal_lattice_vectors
        path: list of list of list of real
            High-symmetry path to be plotted on the Brillouin zone
            Follows the conventions of PhonoPy and pathsLabels.py: 
                - First level: list of connected path segments
                - Second level: list of points that mark path segments
                - Third level: direct coordinates of points
            Default: no path
        path_labels: list of str
            List of names of the edge points to be plotted on the path
            Default: no labels
        label_shifts: list of list of real
            Three-dimensional shifts of the path labels on the plot,
            to reduce overlap with the special points
            Default: all zero, no shifts
        view_angles: tuple of real
            Elevation, azimuth, and roll angles of the plot, in degrees
            Default: Matplotlib default view angles
        save_filename: str
            Save the figure with the given filename, in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        save_bbox_extents: list of list of real
            Extents of the bounding box used to save the figure, in the
            format [[xmin, ymin], [xmax, ymax]]. 
            Default: tight bbox, but this wastes a lot of space
        quiver_plot: np.array of real
            shape (:, 3, 3)
            Data for additional vectors to draw on the plot
                - quiver_plot[...,0]: Origins of the vectors
                - quiver_plot[...,1]: Components of the vectors
                - quiver_plot[...,2]: Coordinates for quiver_labels
            Use an empty list for no vectors
            Default: data for reciprocal lattice vectors b_1, b_2, b_3
        quiver_labels: list of str
            Labels for the additional vectors in quiver_plot
            Default: b_1, b_2, b_3
        visible_linestyle: dict
            Linestyle parameters for lines that are visible under
            the given view angles
            Default: dict(color="black", linestyle="solid", 
                linewidth=1.5)
        invisible_linestyle: dict
            Linestyle parameters for lines that are not visible under
            the given view angles
            Default: dict(color="black", linestyle="dashed", 
                linewidth=1.0)
        path_linestyle: dict
            Linestyle parameters for the high-symmetry path
            Default: dict(marker='o', linestyle="solid", color='red', 
                linewidth=2.0)
        label_style: dict
            Font style parameters for the labels of the high-symmetry
            path
            Default: dict(color="black", fontsize=14)
        quiver_style: dict
            Linestyle parameters for the additional vectors plotted
            using quiver_plot
            Default: dict(color="black", linestyle="solid", 
                linewidth=1.0)
        quiver_label_style:
            Font style parameters for the labels of the additional
            vectors
            Default: dict(color="black", fontsize=14)
        """
        
        if reciprocal_lattice_vectors is None:
            reciprocal_lattice_vectors = self.get_reciprocal_lattice_vectors()
        if visible_linestyle is None:
            visible_linestyle = dict(color="black", linestyle="solid", 
                                     linewidth=1.5)
        if invisible_linestyle is None:
            invisible_linestyle = dict(color="black", linestyle="dashed", 
                                       linewidth=1.0)
        if path_linestyle is None:
            path_linestyle = dict(marker='o', linestyle="solid", color='red', 
                                  linewidth=2.0)
        if quiver_style is None:
            quiver_style = dict(color="black", linestyle="solid", 
                                linewidth=1.0)
        if label_style is None:
            label_style = dict(color="black", fontsize=14)
        if quiver_plot is None:
            quiver_plot = np.empty((len(reciprocal_lattice_vectors), 
                                    self.num_dimensions, 3))
            # The locations where to plot the vectors
            quiver_plot[...,0] = 0.5*np.eye(len(reciprocal_lattice_vectors))
            # The actual vectors to plot
            quiver_plot[...,1] = 0.2*np.eye(len(reciprocal_lattice_vectors))
            # The locations to plot the labels
            quiver_plot[...,2] = 0.8*np.eye(len(reciprocal_lattice_vectors))
        if quiver_labels is None:
            quiver_labels = ["$\\mathbf{b}_1$", "$\\mathbf{b}_2$", 
                             "$\\mathbf{b}_3$"]
        if quiver_label_style is None:
            quiver_label_style = dict(color="black", fontsize=14, 
                                      horizontalalignment="center", 
                                      verticalalignment="center")
        if label_shifts is None:
            label_shifts = np.zeros((len(path_labels), self.num_dimensions))
        
        miller_indices, corners, edges, edge_planes = \
            self.get_Brillouin_boundary(reciprocal_lattice_vectors)
        edge_planes_cart = edge_planes @ reciprocal_lattice_vectors
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if view_angles is None:
            view_angles = (ax.elev, ax.azim, ax.roll)
            
        # Determine which edges are visible, given the current view angles
        theta = 0.5*np.pi - view_angles[0]*np.pi/180
        phi = view_angles[1]*np.pi/180
        view_direction = np.array([np.sin(theta)*np.cos(phi), 
                                   np.sin(theta)*np.sin(phi), np.cos(theta)])
        # A plane is visible if its normal vector aligns with the view 
        # direction (n.v > 0), works because BZ is convex
        # An edge is visible if either of its two planes is visible
        edges_visible = np.any(edge_planes_cart @ view_direction > 0, axis=1)
            
        # Get the unique path labels
        path_points = [point for points in path for point in points]
        unique_points = []
        unique_labels = []
        unique_label_shifts = []
        for point, label, label_shift in zip(path_points, path_labels, 
                                             label_shifts):
            accept_point = True
            for point2 in unique_points:
                if point == point2:
                    accept_point = False
            if accept_point:
                unique_points.append(point)
                unique_labels.append(label)
                unique_label_shifts.append(label_shift)
        unique_points = np.array(unique_points)
        unique_label_shifts = np.array(unique_label_shifts)
        
        # Draw the wireframe outline
        for edge, visible in zip(edges, edges_visible):
            if visible:
                ax.plot(edge[:,0], edge[:,1], edge[:,2], **visible_linestyle)
            else:
                ax.plot(edge[:,0], edge[:,1], edge[:,2], **invisible_linestyle)
        
        # Draw the path
        for mini_path in path:
            points = np.array(mini_path) @ reciprocal_lattice_vectors
            ax.plot(points[:,0], points[:,1], points[:,2], **path_linestyle)
        for point, label, shift in zip(unique_points, unique_labels, 
                                       unique_label_shifts):
            point_cart = (point + shift) @ reciprocal_lattice_vectors
            ax.text(point_cart[0], point_cart[1], point_cart[2], label, 
                    **label_style)
            
        # Draw the reciprocal basis vectors
        vec_origins = quiver_plot[...,0] @ reciprocal_lattice_vectors
        vecs = quiver_plot[...,1] @ reciprocal_lattice_vectors
        label_positions = quiver_plot[...,2] @ reciprocal_lattice_vectors
        ax.quiver(vec_origins[:,0], vec_origins[:,1], vec_origins[:,2], 
                  vecs[:,0], vecs[:,1], vecs[:,2],
                  length=1, arrow_length_ratio=0.3, **quiver_style)
        for point, label in zip(label_positions, quiver_labels):
            ax.text(point[0], point[1], point[2], label, **quiver_label_style)

        # Create cubic bounding box to simulate equal aspect ratio
        data_limits = np.array([[corners[:,0].max(), corners[:,0].min()], 
                                [corners[:,1].max(), corners[:,1].min()], 
                                [corners[:,2].max(), corners[:,2].min()]])
        max_range = (data_limits[:,0]-data_limits[:,1]).max()
        average = np.mean(data_limits, axis=1)
        cube_corners = 0.5*max_range*np.reshape(np.mgrid[-1:2:2,-1:2:2,-1:2:2],
                                                (3,8)).T + average
        ax.plot(cube_corners[:,0], cube_corners[:,1], cube_corners[:,2],
                linestyle='None')
        ax.set_box_aspect([1,1,1])
        ax.grid(False)
        plt.axis('off')
        ax.view_init(elev=view_angles[0], azim=view_angles[1], 
                     roll=view_angles[2])
        
        fig.tight_layout()    
        fig.show()
        
        if save_filename is not None:
            if save_bbox_extents is None:
                save_bbox = "tight"
                create_path(save_filename)
                fig.savefig(save_filename+".pdf", bbox_inches=save_bbox)
            else:
                save_bbox = matplotlib.transforms.Bbox(save_bbox_extents)
                create_path(save_filename)
                fig.savefig(save_filename+".pdf", bbox_inches=save_bbox)
                xy = save_bbox_extents[0]
                width = save_bbox_extents[1][0]-xy[0]
                height = save_bbox_extents[1][1]-xy[1]
                fig.patches.extend(
                    [plt.Rectangle(xy, width, height, fill=False, color='b', 
                                   zorder=1000, transform=fig.dpi_scale_trans, 
                                   figure=fig)])
        

class PhonopyMeshCalculation(PhonopyCalculation):
    """ Class to load PhonoPy mesh calculations
    
    Can calculate the phonon density of states from the PhonoPy data
    Has the same attributes as PhonopyCalculation
    """

    def __init__(self, yaml_filename):
        """ PhonopyMeshCalculation(yaml_filename, born_filename)

        Arguments
        ---------
        yaml_filename: string
            .yaml file exported by PhonoPy
        born_filename: string
            BORN file that contains the Born effective charge tensors
            and dielectric tensors.
            Important: this function expects a BORN file with a line
            for each atom, since no symmetry is implemented. PhonoPy
            exports a BORN file with less lines based on symmetry,
            which is not compatible with this code.
        """
        super().__init__(yaml_filename)
        
    def calculate_DOS(self, unit="THz", sigma=None, omega=None, num_omegas=501):
        """ Calculate the phonon DOS with the smearing method

        Arguments
        ---------
        unit: str
            Phonon frequency unit, default "THz"
        sigma: real
            Gaussian smearing used in the smearing method, in units
            of the phonon frequency
            Default: 0.01*(omega_max - omega_min)
        omega: np.array of real
            Frequencies at which the phonon DOS is to be evaluated
            Default: rounded range of num_omegas points between the
            minimum and maximum phonon frequency
        num_omegas: int
            Number of points used in the default frequency range
            Only used when omega is None
            Default: 501

        Returns
        -------
        omega: np.array of real
            Frequencies at which the phonon DOS is evaluated
        DOS: np.array of real
            Phonon density of states at frequencies omega

        """
        phonon_freqs = self.get_frequencies(unit)
        omega_max = np.max(phonon_freqs)
        omega_min = np.min(phonon_freqs)
        if not omega:
            # Set the default range of omega if none is provided
            rounding_scale = 10**np.floor(np.log10(abs(omega_max)))
            omega_min_scale = np.trunc(omega_min/rounding_scale)*rounding_scale 
            omega_max_scale = np.ceil(omega_max/rounding_scale)*rounding_scale 
            omega = np.linspace(omega_min_scale, omega_max_scale, 
                                num=num_omegas, endpoint=True)
        if not sigma:
            # Set the default value of sigma if none is provided
            sigma = 0.01*(omega_max - omega_min)
        two_sigma_squared = 2*sigma*sigma
        # Define the smeared delta function:
        delta_smeared = lambda x : \
            np.exp(- x*x/two_sigma_squared)/np.sqrt(np.pi*two_sigma_squared)
        # Define the DOS as calculated with the trapezoid rule:
        DOS_func = lambda x : \
            np.sum(self.weights*np.sum(delta_smeared(x-phonon_freqs),axis=1))/\
                np.sum(self.weights)
        #Iterate DOS_func over all elements in omega:
        with np.nditer([omega, None], op_dtypes=float, flags=['buffered']) \
            as iterator:
            for freq, output in iterator:
                output[...] = DOS_func(freq)
            DOS =  iterator.operands[1]
        return omega, DOS
    
        
class PhonopyBandCalculation(PhonopyCalculation):
    """ Class to load PhonoPy band calculations
    
    Can plot the phonon density of states along a high-symmetry path
    Has the same attributes as PhonopyCalculation
    """
        
    def __init__(self, yaml_filename):
        """ PhonopyBandCalculation(yaml_filename, born_filename)

        Arguments
        ---------
        yaml_filename: string
            .yaml file exported by PhonoPy
        born_filename: string
            BORN file that contains the Born effective charge tensors
            and dielectric tensors.
            Important: this function expects a BORN file with a line
            for each atom, since no symmetry is implemented. PhonoPy
            exports a BORN file with less lines based on symmetry,
            which is not compatible with this code.
        """
        super().__init__(yaml_filename)
        
    def get_xaxis_labels(self):
        """ Parse labels and path distances to plot-ready format

        Returns
        -------
        xaxis_ticks: np.array of real
            Locations to plot the labels on the x-axis
        xaxis_labels: list of str
            Labels of high-symmetry points to plot on the x-axis
        """
        # Generate the labels
        xaxis_labels = []
        xaxis_labels.append(self.labels[0][0])
        for i in range(len(self.labels)-1):
            label1 = self.labels[i][1]
            label2 = self.labels[i+1][0]
            if label1 == label2:
                xaxis_labels.append(label1)
            else:
                xaxis_labels.append(label1+"|"+label2)
        xaxis_labels.append(self.labels[-1][1])
        # Generate the positions of the labels along the x-axis
        indices = np.cumsum([-1]+self.segment_nqpoint)
        indices[0]=0
        xaxis_ticks = self.distances[indices]
                
        return xaxis_ticks, xaxis_labels
    
    def plot(self, unit="THz", title="Phonon dispersion", save_filename=None, 
             plot_range=None, text_sizes=(13, 15, 16)):
        """ Plot the band structure in a figure

        Arguments
        ---------
        unit: str
            Phonon frequency unit, default "THz"
        title: str
            Title of the plot
            Default: "Phonon dispersion"
        save_filename: str
            Save the figure with the given filename, in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        plot_range: tuple of 2 reals
            Minimum and maximum limits of the figure y-axis
            Default: rounded range of num_omegas points between the
            minimum and maximum phonon frequency
        text_sizes: tuple of 3 ints
            Font sizes for small, medium and large text in the figure
                -Small text: axis ticks
                -Medium text: axis labels
                -Large text: title
            Default: (13, 15, 16)
        
        Returns
        -------
        fig: figure handle
        ax: axis handle
        plot_handle: handle to the plot data

        """
        plot_color = "blue"
        plot_linestyle = "solid"
        
        qs = self.distances
        omegas = self.get_frequencies(unit)
        
        fig, ax = plt.subplots()
        plot_handle = ax.plot(qs, omegas, color=plot_color, 
                              linestyle=plot_linestyle)
        
        xaxis_ticks, xaxis_labels = self.get_xaxis_labels()
        
        ax.set_title(title, size=text_sizes[2])
        ax.set_ylabel('Phonon frequencies ('+str(unit)+')', 
                      fontsize=text_sizes[1])
        ax.set_xticks(xaxis_ticks, xaxis_labels)
        ax.tick_params(axis='both', labelsize=text_sizes[0])
        ax.set_xlim(np.min(qs),np.max(qs))
        
        if plot_range is None:
            omega_min = np.min(omegas)
            omega_max = np.max(omegas)
            if omega_min < -self.convert_units(0.1, from_unit="THz", 
                                               to_unit=unit):
                # Include the unstable phonon modes in the plot
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max)
            else:
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max, clamp_min=0)
        else:
            omega_min_scale = plot_range[0]
            omega_max_scale = plot_range[1]
        ax.set_ylim(omega_min_scale, omega_max_scale)
        
        # Plot major axis ticks
        for tick in xaxis_ticks:
            plt.axvline(x=tick, color='black', linewidth=1.0)
        
        fig.tight_layout()
        fig.show()
        if save_filename is not None:
            create_path(save_filename)
            plt.savefig(save_filename+".pdf")
        return fig, ax, plot_handle

class PhonopyCommensurateCalculation(PhonopyCalculation):
    """ Class to process PhonoPy calculation at commensurate points

    This class reads in the PhonoPy data of a calculation at only
    the commensurate points in the Brillouin zone. It has several
    functionalities of PhonoPy reimplemented, such as Fourier
    interpolation and plotting phonon bands. It also allows for the
    calculation of longitudinal/transverse and acoustic/optical
    weights.
    
    Attributes
    ----------
    supercell_size: np.array of int
        shape (3,)
    reciprocal_lattice_vectors: np.array of real
        shape (3,3), units of inverse Angstroms
    lattice_vectors: np.array of real
        shape (3,3), units of Angstroms
    unitcell_volume: real
        units of cubic Angstroms
    num_dimensions: int
        Equal to 3
    natom: int
    numqpoints: int
    labels: list of (list of str) if band calculation, else None
    segment_nqpoint: list of int if band calculation, else None
    numbands: int
        Equal to 3*natom
    qpoints: np.array of real
        shape (numqpoints, 3), in direct coordinates
    distances: np.array of real
        shape (numqpoints,)
    weights: np.array of int
        shape (numqpoints,)
    frequencies: np.array of real
        shape (numqpoints, numbands), stored as cycle frequencies in THz
    eigenvectors: np.array of complex, 
        shape (numqpoints, numbands, natom, 3)
    atom_names: np.array of str
        shape (natom,)
    atom_masses: np.array of real
        shape (natom,), in atomic mass units
    atom_positions: np.array of real
        shape (natom, 3), in direct coordinates
    born_is_set: bool
        True if born_filename is read
    born_charges: np.array of real
        shape (natom, 3, 3)
    dielectric_tensor: np.array of real
        shape (3,3)
    Gpoints: np.array of int
        shape (:,3), used for G-sums, stored in direct coordinates
    nac_q_direction: np.array of real
        shape (3,)
    nac_G_cutoff: real
    nac_lambda: real
    optimal_sc_vectors: list of list of list of np.array of int
        First list index: lattice vector L
        Second list index: atom index k'
        Third list index: atom index k
        np.array shapes: (:,3)

    """
    def __init__(self, yaml_filename, born_filename=None, 
                 nac_q_direction=np.array([1.0,0.0,0.0]), nac_G_cutoff=None, 
                 nac_lambda=None):
        """PhonopyCommensurateCalculation(yaml_filename, born_filename)

        Arguments
        ---------
        yaml_filename: string
            .yaml file exported by PhonoPy
        born_filename: string
            BORN file that contains the Born effective charge tensors
            and dielectric tensors.
            Important: this function expects a BORN file with a line
            for each atom, since no symmetry is implemented. PhonoPy
            exports a BORN file with less lines based on symmetry,
            which is not compatible with this code.
        nac_q_direction: np.array of real
            In case the dynamical matrix is not analytic at Gamma,
            store the values with limiting direction nac_q_direction
            Default: np.array([1.0, 0.0, 0.0])
        nac_G_cutoff: real
            Cutoff radius for reciprocal lattice sums
            Default: Same as PhonoPy, chosen to include approximately 
            300 reciprocal lattice points
        nac_lambda: real
            Exponential decay for reciprocal lattice sums
            Default: Same as PhonoPy, based on self.dielectric_tensor
            and self.nac_G_cutoff

        """
        super().__init__(yaml_filename, born_filename)
        self.optimal_sc_vectors = self.get_optimal_sc_vectors()
        self.set_nac_q_direction(nac_q_direction)
        self.set_nac_G_cutoff(nac_G_cutoff)
        self.set_nac_lambda(nac_lambda)
        
        
    def set_nac_q_direction(self, nac_q_direction):
        """ Set nac_q_direction to a different value """
        if nac_q_direction is not None:
            self.nac_q_direction = nac_q_direction
        
    def set_nac_G_cutoff(self, nac_G_cutoff):
        """ Set nac_G_cutoff to a different value

        Automatically recalculates self.Gpoints. If input is None, 
        set to the default value used by PhonoPy. This default is
        nac_G_cutoff = (3*300/(4*np.pi*self.unitcell_volume))**(1.0/3).
        """
        if nac_G_cutoff is None:
            num_Gs = 300
            nac_G_cutoff = (3*num_Gs / (4*np.pi*self.unitcell_volume))**(1.0/3)
        self.nac_G_cutoff = nac_G_cutoff
        self.Gpoints = self.get_Gpoints(nac_G_cutoff, include_zero=False)
        
    def set_nac_lambda(self, nac_lambda):
        """ Set nac_G_lambda to a different value
        
        If input is None, set to the default value used by PhonoPy,
        which is nac_G_cutoff*np.sqrt(0.025*eps_avg/log(10))
        """
        if nac_lambda is None:
            log_cutoff = -10*np.log(10)
            if self.dielectric_tensor is None:
                eps_avg = 1.0
            else:
                eps_avg = np.trace(self.dielectric_tensor)/self.num_dimensions
            nac_lambda = self.nac_G_cutoff*np.sqrt(-0.25*eps_avg/log_cutoff)
        self.nac_lambda = nac_lambda
        
    def get_optimal_sc_vectors(self, distance_tolerance = 1e-10):
        """ Find supercell vectors T that minimize |L+tau_k'-tau_k+T|

        For every atomic distance vector L+tau_k'-tau_k, find the
        supercell vector that minimizes the distance |L+tau_k'-tau_k+T|.
        These supercell vectors are used for the Fourier interpolation
        algorithm, but are not important for the user

        Arguments
        ---------
        distance_tolerance: real
            Tolerance to decide whether two points are equally distant
            Default: 1e-10
        
        Returns
        -------
        optimal_sc_vectors: list of list of list of np.array of int
            Nested list of optimal supercell vectors, for every atomic
            distance vector L+tau_k'-tau_k
                - First list index: lattice vector L
                - Second list index: atom index k'
                - Third list index: atom index k
            Each list element is a np.array of int with shape (:,3)
            that contains all the supercell vectors that minimize
            |L+tau_k'-tau_k+T| in direct coordinates

        """
        metric = self.lattice_vectors @ self.lattice_vectors.T
        norm = lambda vec: np.sqrt(vec @ metric @ vec.T)
        sc_weights = [-1, 0, 1]
        Tpoint_candidates = np.zeros((len(sc_weights)**self.num_dimensions, 
                                      self.num_dimensions), dtype=int)
        for index1, i in enumerate(sc_weights):
            for index2, j in enumerate(sc_weights):
                for index3, k in enumerate(sc_weights):
                    Tpoint_candidates[index3 + len(sc_weights)*\
                                      (index2+ len(sc_weights)*index1)] \
                            = np.array([k,j,i])*self.supercell_size
        optimal_sc_vectors = []
        for lpoint in self.get_lpoints():
            Tpoint_list_2 = []
            for tau_k2 in self.get_atom_positions():
                Tpoint_list_3 = []
                for tau_k1 in self.get_atom_positions():
                    distances = np.array([norm(lpoint+tau_k2-tau_k1+Tpoint) 
                                          for Tpoint in Tpoint_candidates])
                    min_distance = distances.min()
                    optimal_Tpoints = \
                        Tpoint_candidates[np.abs(distances-min_distance) \
                                          < distance_tolerance]
                    Tpoint_list_3.append(optimal_Tpoints)
                Tpoint_list_2.append(Tpoint_list_3)
            optimal_sc_vectors.append(Tpoint_list_2)
        return optimal_sc_vectors
    
    def set_optimal_sc_vectors(self, distance_tolerance = 1e-10):
        """ Set optimal supercell vectors in object """
        self.optimal_sc_vectors=self.get_optimal_sc_vectors(distance_tolerance)
                
    def fourier_interpolate(self, q, quantity=None, unit="THz", 
                            convention="c-type", include_nac="None", 
                            nac_q_direction=None, nac_G_cutoff=None, 
                            nac_lambda=None):
        """ Fourier interpolate dynamical matrix or similar quantity

        It is assumed that the quantity is given in the convention that
        matches the input convention. If no quantity is given, the
        dynamical matrix will be calculated and interpolated
        Warning: The NAC correction is only implemented for the
        dynamical matrix. For Fourier interpolation of other 
        quantities, we automatically choose include_nac="None" to avoid
        incorrect results.

        Arguments
        ---------
        q: np.array of real, shape (3,) or (:,3)
            q-point, or array of q-points, in which to evaluate the
            Fourier-interpolated dynamical matrix
        quantity: np.array of complex
            shape (numqpoints, numbands, numbands)
            A quantity of the same shape as the dynamical matrices 
            evaluated at self.qpoints, the commensurate q-points
            Default: the dynamical matrices evaluated at self.qpoints
        unit: str
            Unit for the phonon frequencies
            Default: "THz"
        convention: str
            c-type or d-type convention for the dynamical matrix
            Default: c-type
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        nac_q_direction: np.array of real
            In case the dynamical matrix is not analytic at Gamma,
            store the values with limiting direction nac_q_direction
            Default: np.array([1.0, 0.0, 0.0])
        nac_G_cutoff: real
            Cutoff radius for reciprocal lattice sums
            Default: Same as PhonoPy, chosen to include approximately 
            300 reciprocal lattice points
        nac_lambda: real
            Exponential decay for reciprocal lattice sums
            Default: Same as PhonoPy, based on self.dielectric_tensor
            and self.nac_G_cutoff

        Returns
        -------
        interpolated_quantity: np.array of complex
            shape (len(q), numbands, numbands),
            or (numbands, numbands) if q is a single q-point
            Interpolated quantity evaluated at the desired q-points,
            usually the dynamical matrix or its derivative

        """
        
        if quantity is None:
            quantity = self.get_dynamical_matrices(unit=unit, 
                                                   convention="d-type")
        else:
            include_nac="None"
            if convention=="c-type":
                quantity = quantity*self.get_c_to_d_factors(self.get_qpoints())
            
        self.set_nac_q_direction(nac_q_direction)
        self.set_nac_G_cutoff(nac_G_cutoff)
        self.set_nac_lambda(nac_lambda)
        
        # Ensure any element of q is never smaller than 1e-12
        q_clean = (np.abs(q)+1e-12)*(2*np.heaviside(q, 1)-1)

        # Ensure that the output has the desired dimension
        return_1D = False
        if len(q_clean.shape) < 2:
            return_1D = True
            q_clean = np.array([q_clean])  # Ensure q_clean is a 2D array
        
        # Get the force constants, or its equivalent for the desired quantity
        interpolated_quantities = np.empty((len(q_clean), self.numbands, 
                                            self.numbands), dtype=np.complex_)     
        force_constants = self.get_force_constants(quantity=quantity, unit=unit,
                                                   include_nac=include_nac)
        
        # Make the matrix of Fourier factors that will transform the
        # force constants to the dynamical matrix:
        T_opts = self.optimal_sc_vectors
        fourier_factors = np.zeros((len(q_clean), len(self.get_lpoints()), 
                                    self.numbands, self.numbands), 
                                   dtype=np.complex_)
        for index2, lpoint in enumerate(self.get_lpoints()):
            for index3, tau_k2 in enumerate(self.get_atom_positions()):
                for index4, tau_k1 in enumerate(self.get_atom_positions()):
                    r_tots = lpoint + tau_k2 - tau_k1 + \
                        T_opts[index2][index3][index4]
                    fourier_factors[:, index2, self.num_dimensions*index4:\
                                    (self.num_dimensions*(index4+1)), 
                                    self.num_dimensions*index3:\
                                    (self.num_dimensions*(index3+1))] \
                        = np.reshape(
                            np.mean(np.exp(2*np.pi*1j* q_clean @ r_tots.T ),1),
                            (-1,1,1))
            
        # Make the dynamical matrix in the c-type convention
        quantity_proposal = np.sum(fourier_factors * force_constants, 1)
        match include_nac:
            case "Wang":
                wangMatrix = np.mean(fourier_factors, 1)*\
                    self.get_nac_dynamical_matrix_Wang(q_clean, \
                            unit=unit, convention="c-type",
                            nac_q_direction=self.nac_q_direction)
                quantity_ctype = (wangMatrix + 
                    0.5*(quantity_proposal+
                         np.swapaxes(quantity_proposal,1,2).conj()))
            case "Gonze":
                gonzeMatrix = self.get_nac_dynamical_matrix_Gonze(q_clean, \
                    unit=unit, convention="c-type", 
                    nac_q_direction=self.nac_q_direction)
                quantity_ctype = (gonzeMatrix + 
                    0.5*(quantity_proposal+
                         np.swapaxes(quantity_proposal,1,2).conj()))
            case _:
                #No NAC correction
                quantity_ctype = 0.5*(quantity_proposal+
                                      np.swapaxes(quantity_proposal,1,2).conj())
        
        # Transform the dynamical matrix to the d-type convention if necessary
        match convention:
            case "c-type":
                interpolated_quantities = quantity_ctype
            case "d-type":
                interpolated_quantities = quantity_ctype*\
                    self.get_c_to_d_factors(q)
            case _:
                raise ValueError("convention must be c-type or d-type")
        if return_1D:
            # If there was only one q-point, return just that dynamical matrix
            return interpolated_quantities[0]
        else:
            # Otherwise, return an array of dynamical matrices
            return interpolated_quantities        
    
    def get_force_constants(self, quantity=None, unit="THz", 
                            include_nac="None"):
        """ Calculate short-range force constants

        This function returns the short-range force constants, or what
        are considered the short-range force constants for the method 
        and quantity we use. The force constants are already divided
        by the inverse square root of the atom masses

        Arguments
        ---------
        quantity: np.array of complex
            shape (numqpoints, numbands, numbands)
            A quantity of the same shape as the dynamical matrices 
            evaluated at self.qpoints, the commensurate q-points
            Default: the dynamical matrices evaluated at self.qpoints
        unit: str
            Unit for the phonon frequencies
            Default: "THz"
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"

        Returns
        -------
        force_constants: np.array of real
            shape (numqpoints, numbands, numbands)

        """

        if quantity is None:
            quantity = self.get_dynamical_matrices(unit=unit, 
                                                   convention="d-type")
        
        match include_nac:
            case "Gonze":
                # Use get_nac_dynamical_matrix_Gonze with nac_q_direction=None
                # to get only the analytic part of the dynamical matrix
                dyn_long_range =\
                    self.get_nac_dynamical_matrix_Gonze(self.qpoints, unit=unit,
                                                        convention="d-type",
                                                        nac_q_direction=None)
            case _:
                # Both in Wang's method and in the case of no NAC correction,
                # there are no long-range force constants to subtract
                dyn_long_range = 0.
        dyn_matrix = quantity - dyn_long_range
        fourier_matrix = np.exp(-2*np.pi*1j* self.get_qpoints() @\
                                self.get_lpoints().T )
        return np.real(np.tensordot(fourier_matrix, dyn_matrix, 1)/ \
                       self.numqpoints)
    
    def get_nac_dynamical_matrix_Wang(self, q, unit="THz", convention="d-type", 
                                      nac_q_direction=np.array([1.0,0.0,0.0])):
        """ Get NAC correction with Wang's method

        Based on the method detailed in:
        Y Wang et al., J. Phys.: Condens. Matter 22 202201 (2010)

        This method is faster than Gonze's method but sometimes gives
        unphysical unstable phonon modes. Gonze's method is recommended

        Arguments
        ---------
        q: np.array of real, shape (3,) or (:,3)
            q-point, or array of q-points, in which to evaluate the
            Fourier-interpolated dynamical matrix
        unit: str
            Unit for the phonon frequencies
            Default: "THz"
        convention: str
            c-type or d-type convention for the dynamical matrix
            Default: c-type
        nac_q_direction: np.array of real
            In case the dynamical matrix is not analytic at Gamma,
            store the values with limiting direction nac_q_direction
            Default: np.array([1.0, 0.0, 0.0])

        Returns
        -------
        dynamical matrix: np.array of complex
            shape (len(q), numbands, numbands),
            or (numbands, numbands) if q is a single q-point
            Interpolated dynamical matrix evaluated at the desired
            q-points

        Raises
        ------
        ValueError
            when calling this function without loading a BORN file

        """
        if not self.born_is_set:
            raise ValueError("""BORN file must be read to calculate 
                             the non-analytic contribution""")
        # Ensure that the output has the desired dimension
        return_1D = False
        if len(q.shape) < 2:
            return_1D = True
            q = np.array([q])  # Ensure q is a 2D array at least
        for zero_index in np.nonzero(np.linalg.norm(q, ord=2, axis=1)<1e-10)[0]:
            # In the Gamma point, use a tiny vector in direction nac_q_direction
            q[zero_index] = 1e-12*nac_q_direction
        masses = np.diag(self.get_mass_matrix())
        inv_mass_matrix = 1/np.sqrt(masses.reshape(1, self.numbands)* \
                                    masses.reshape(self.numbands, 1))
        born_charges_3x3N = np.reshape(self.born_charges.T,
                                       (self.num_dimensions,self.numbands), 
                                       order='F')
            
        outer_product = lambda A: A[...,:,None]*A[...,None,:]
        
        #Define the basic expression for the NAC part of the dynamical matrix:
        Q = q @ self.get_reciprocal_lattice_vectors()        
        dynmat_proposal = (
            # Prefactor to get the units right:    
            (1.7459144492158638e+30/self.unitcell_volume)*\
                self.convert_units(1, from_unit="rad/s", to_unit=unit)**2
            # Multiply by the inverse mass matrix:
            * inv_mass_matrix
            # Numerator: Z.Q x Q.Z
            * outer_product(Q @ born_charges_3x3N)
            # Denominator: Q.eps.Q
            / np.sum((Q @ self.dielectric_tensor) * Q, -1)[...,None,None]
            )
        
        match convention:
            case "c-type":
                dynmats = dynmat_proposal.astype(np.complex_)
            case "d-type":
                dynmats = dynmat_proposal * self.get_c_to_d_factors(q)
            case _:
                raise ValueError("convention must be c-type or d-type")
        if return_1D:
            return dynmats[0]
        else:
            return dynmats  
    
    def get_nac_dynamical_matrix_Gonze(self, q, unit="THz", convention="d-type", 
                                       nac_q_direction=np.array([1.0,0.0,0.0])):
        """ Get NAC correction with Gonze's method

        Based on the method detailed in:
        X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)

        Set nac_q_direction = None to return only the analytic part 
        at q=0

        Arguments
        ---------
        q: np.array of real, shape (3,) or (:,3)
            q-point, or array of q-points, in which to evaluate the
            Fourier-interpolated dynamical matrix
        unit: str
            Unit for the phonon frequencies
            Default: "THz"
        convention: str
            c-type or d-type convention for the dynamical matrix
            Default: c-type
        nac_q_direction: np.array of real
            In case the dynamical matrix is not analytic at Gamma,
            store the values with limiting direction nac_q_direction
            Default: np.array([1.0, 0.0, 0.0])

        Returns
        -------
        dynamical matrix: np.array of complex
            shape (len(q), numbands, numbands),
            or (numbands, numbands) if q is a single q-point
            Interpolated dynamical matrix evaluated at the desired
            q-points

        Raises
        ------
        ValueError
            when calling this function without loading a BORN file

        """
        
        no_nac_flag = nac_q_direction is None
        
        if not self.born_is_set:
            raise ValueError("""BORN file must be read to calculate 
                             the non-analytic contribution""")
        # Ensure that the output has the desired dimension
        return_1D = False
        if len(q.shape) < 2:
            return_1D = True
            q = np.array([q])  # Ensure q is a 2D array at least
        if not no_nac_flag: 
            # In the Gamma point, use a tiny vector in direction nac_q_direction
            for zero_index in np.nonzero(np.linalg.norm(q, ord=2, axis=1)\
                                          < 1e-10)[0]:
                    q[zero_index] = 1e-12*nac_q_direction
        dynmats = np.empty((len(q), self.numbands, self.numbands), 
                           dtype=np.complex_)   
        masses = np.diag(self.get_mass_matrix())
        inv_mass_matrix = 1/np.sqrt(masses.reshape(1, self.numbands)* \
                                    masses.reshape(self.numbands, 1))
        born_charges_3x3N = np.reshape(self.born_charges.T,
                                       (self.num_dimensions,self.numbands),
                                       order='F')
        tauk_difference_cart = self.get_tauk_difference() @ self.lattice_vectors
        
        
        q_cart = q @ self.get_reciprocal_lattice_vectors()
        Gs_cart = self.Gpoints @ self.get_reciprocal_lattice_vectors()
        
        
        # Define the basic expression for the NAC part of the dynamical matrix
        outer_product = lambda A: A[...,:,None]*A[...,None,:]
        QepsQ_function = lambda x: np.exp(-x/(4*self.nac_lambda**2))/x
        dyn_nac_Q = lambda Q_cart: (
            # Prefactor to get the units right:    
            1.7459144492158638e+30/self.unitcell_volume * \
                self.convert_units(1, from_unit="rad/s", to_unit=unit)**2
            # Numerator: Z.Q x Q.Z
            * outer_product(Q_cart @ born_charges_3x3N)
            # Denominator: Q.eps.Q, with exponential damping
            * QepsQ_function(np.sum((Q_cart @ self.dielectric_tensor) * \
                                    Q_cart, -1))[...,None,None]
            # Fourier factors in the d-type convention
            * np.exp(2*np.pi*1j* \
                     np.dot(Q_cart, np.swapaxes(tauk_difference_cart, 1, 2)))
            )
        
        # Make the traceless tensor:
        # dyn_trace = delta_{k,k'} sum_k'' sum_G C_{k,k''}(G)
        tensor_trace = np.sum(np.reshape(np.sum(dyn_nac_Q(Gs_cart), axis=0),
                            (self.numbands,self.natom,self.num_dimensions)), 
                            axis=1)
        dyn_trace = np.zeros((self.numbands, self.numbands), dtype=np.complex_)
        for k in range(self.natom):
            to_slice = slice(self.num_dimensions*k, self.num_dimensions*(k+1), 
                             1)
            dyn_trace[to_slice, to_slice] = tensor_trace[to_slice, :]
        
        dynmat_proposal = np.zeros((len(q), self.numbands, self.numbands), 
                                   dtype=np.complex_)
        for index, q_point in enumerate(q_cart):
            if np.linalg.norm(q_point) < 1e-10 and no_nac_flag:
                G0_contribution = 0.
            else:
                G0_contribution = dyn_nac_Q(np.array([q_point]))
            dynmat_proposal[index, ...] = \
                inv_mass_matrix*(G0_contribution+\
                                 np.sum(dyn_nac_Q(q_point+Gs_cart),0)-dyn_trace)
        match convention:
            case "c-type":
                dynmats = dynmat_proposal * self.get_d_to_c_factors(q)
            case "d-type":
                dynmats = dynmat_proposal
            case _:
                raise ValueError("convention must be c-type or d-type")
        if return_1D:
            return dynmats[0]
        else:
            return dynmats  
    
    def get_freqs_eigvecs_interpolated(self, q, unit="THz", convention="c-type",
                                        include_nac="None", clean_value=None):
        """ Calculate interpolated phonon frequencies and eigenvectors

        Arguments
        ---------
        q: np.array of real, shape (3,) or (:,3)
            q-point, or array of q-points, in which to evaluate the
            Fourier-interpolated frequencies and eigenvectors
        unit: str
            Unit for the phonon frequencies
            Default: "THz"
        convention: str
            c-type or d-type convention for the phonon eigenvectors
            Default: c-type
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        clean_value: real, or None
            If real, clean frequencies to remove imaginary modes
            and set the minimum frequency value to clean_value
            Default: None

        Returns
        -------
        omegas: np.array of real
            shape (:, numbands)
            Phonon frequencies evaluated at input q

        eigvecs: np.array of complex
            shape (:, numbands, numbands)
            Phonon eigenvectors evaluated at input q
            First index: labels the different bands
            Second index: labels 1x, 1y, 1z, 2x, 2y, 2z, ...
        
        
        """
        # Ensure any element of q is never smaller than 1e-12
        q_clean = (np.abs(q)+1e-12)*(2*np.heaviside(q, 1)-1)  
        # Ensure that the output has the desired dimension
        return_1D = False
        if len(q_clean.shape) < 2:
            return_1D = True
            q_clean = np.array([q_clean])  # Ensure q_clean is a 2D array
        omegas = np.empty((len(q_clean), self.numbands), dtype=float)
        eigvecs = np.empty((len(q_clean), self.numbands, self.numbands), 
                           dtype=np.complex_)
        dynmats = self.fourier_interpolate(q_clean, unit=unit, 
                                           convention=convention, 
                                           include_nac=include_nac)
        for index, dynmat in enumerate(dynmats):
            eigenvalues, eigenvectors = np.linalg.eigh(dynmat)
            omegas_unsorted = np.real(np.sign(eigenvalues)*\
                                      np.sqrt(np.abs(eigenvalues)))
            sort_indices = np.argsort(omegas_unsorted)
            omegas[index, :] = omegas_unsorted[sort_indices]
            eigvecs[index, ...] = eigenvectors[:, sort_indices].T
        if clean_value is not None:
            # Clean frequencies and set the minimum value to clean_value
            omegas = self.get_clean_frequencies(frequencies_to_clean = omegas, 
                                                unit=unit, 
                                                min_value=clean_value)
        if return_1D:
            # If there was only one q-point, return just that set of frequencies
            return omegas[0], eigvecs[0]
        else:
            # Otherwise, return an array of dynamical matrices
            return omegas, eigvecs
        
    def get_label_permutations(self, qs, eigenvectors=None, 
                               jump_indices=np.array([])):
        """ Phonon label permutations for correct crossings in plots

        For a given succession on q-points along a path, 
        gives the permutations of the labels so that the band structure
        plots of the frequencies has the correct crossings. This is
        calculated from the overlap of the corresponding phonon
        eigenvectors.

        Arguments
        ---------
        qs: np.array of real
            shape(:, 3)
            q-points along a path, used to make a phonon band plot
        eigenvectors: np.array of complex
            shape (len(qs), numbands, numbands)
            Default: recalculate eigenvecs for qs from scratch
        jump_indices: list of int
            Indices where the path shows a discontinuous jump,
            as output by self.parse_path

        Returns
        -------
        band_label_permutations: np.array of int (only 0 or 1)
            shape (len(qs), numbands, numbands)
            Stack of permutation matrices
            For each q-point, omega @ band_label_permutation
            gives the correctly permuted phonon frequencies
        
        """
        if eigenvectors is None:
            _, eigenvectors = \
                self.get_freqs_eigvecs_interpolated(qs, convention="c-type", 
                                                    include_nac="Gonze")
        band_label_permutations = np.tile(np.identity(self.numbands, dtype=int),
                                          (len(eigenvectors),1,1))
        for index in range(len(eigenvectors)-1):
            # Exclude overlaps with the gamma point, and with jumps in the path
            if (not any(np.linalg.norm(qs[index:(index+2)], axis=1) < 1e-5) 
                and not index in jump_indices):
                overlap_matrix = np.abs(eigenvectors[index] @ \
                                        eigenvectors[index+1].conj().T)**2
                permutation_matrix = np.zeros((self.numbands, self.numbands), 
                                              dtype=int)
                while (overlap_matrix > -0.5).any():
                    max_value = np.max(overlap_matrix)
                    max_indices = np.where(overlap_matrix==max_value)
                    permutation_matrix[max_indices] = 1
                    overlap_matrix[max_indices[0],:] = -1
                    overlap_matrix[:,max_indices[1]] = -1                    
                permutation_matrix = permutation_matrix.T
                current_permutation = band_label_permutations[index]
                band_label_permutations[index+1:] = \
                    permutation_matrix @ current_permutation
        return band_label_permutations

    
    def get_LATO_weights(self, qs, eigenvectors=None):
        """ Calculate TA, LA, TO, and LO "weights" of phonon bands

        The LATO weights are calculated with the eigenvectors in the
        c-type convention. They are positive and add up to one.

        Arguments
        ---------
        qs: np.array of real
            shape(:, 3)
            q-points at which to calculate the LATO weights
        eigenvectors: np.array of complex
            shape (len(qs), numbands, numbands)
            Default: recalculate eigenvecs for qs from scratch

        Returns
        -------
        weight_matrices: np.array of real
            shape(len(qs), 4, numbands)
            LATO weights for each of the bands
            Second index represents TA, LA, TO, LO in order
        """
        
        if eigenvectors is None:
            _, eigenvectors = \
                self.get_freqs_eigvecs_interpolated(qs, convention="c-type", 
                                                    include_nac="Gonze")
        zero_indices = np.nonzero(np.linalg.norm(qs, ord=2, axis=1) < 1e-10)[0]
        qs[zero_indices] = 1e-12*np.array([1.0, 1.0, 1.0])  # Gamma point
        
        ms = self.get_atom_masses()
        PA = np.sqrt(ms[:,None]*ms)/np.sum(ms)
        PO = np.eye(self.natom) - PA
        q_cart = qs @ self.get_reciprocal_lattice_vectors()
        outer_product = lambda A: A[...,:,None]*A[...,None,:]
        PL = outer_product(q_cart/np.linalg.norm(q_cart, ord=2, axis=-1)[:,None])
        PT = np.eye(self.num_dimensions) - PL
        projection_tensor = np.array([np.kron(PA, PT), np.kron(PA, PL), 
                                      np.kron(PO, PT), np.kron(PO, PL)])
        LATO_weights = np.real((eigenvectors.conj() @ projection_tensor @ \
                                eigenvectors.swapaxes(-1,-2))\
                                .diagonal(axis1=-1,axis2=-2).swapaxes(0,1))
        # Somehow this array is read-only so we have to make it writeable
        LATO_weights.setflags(write=True)
                
        # Separately calculate the weights at Gamma
        weights_gamma = np.zeros((4, self.numbands))
        weights_gamma[0, 0:(self.num_dimensions-1)] = 1
        weights_gamma[1, (self.num_dimensions-1):self.num_dimensions] = 1
        weights_gamma[2, self.num_dimensions:\
                      (self.num_dimensions+\
                       (self.num_dimensions-1)*(self.natom-1))] = 1
        weights_gamma[3, (self.num_dimensions+(self.num_dimensions-1)\
                          *(self.natom-1)):self.num_dimensions*self.natom] =1
        LATO_weights[zero_indices] = weights_gamma

        return LATO_weights
    
    def polaron_ZPR(self, band_mass, lebedev_order=21, include_nac="Gonze",
                    temperature=0):
        """ Calculate weak-coupling polaron ZPR and effective alpha
        
        Calculates the polaron zero-point renormalization (ground
        state energy) based on the phonon frequencies and eigenvectors,
        Born effective charge tensor, dielectric tensor, and an
        effective electron mass. This function implements the 
        weak-coupling formula resulting from the generalized Fröhlich
        Hamiltonian. Also returns an effective Fröhlich alpha, as
        defined in https://doi.org/10.1038/s41524-023-01083-8.

        Arguments
        ---------
        band_mass: real or str
            If real: represents the electron band mass, in units of 
            the bare electron mass
            If str: path to file that contains info about warped
            band masses 
        lebedev_order: int
            Order of the grid used for Lebedev quadrature
            Only used when band_mass is not a file
            Must be an order allowed by scipy.integrate.lebedev_rule
            (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113,
            119, 125, 131)
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "Gonze"
        temperature: real
            Temperature, in Kelvin
            If T>0, this can be used to calculate the band gap
            renormalization at finite temperatures, and the polaron
            lifetime
            Default: 0

        Returns
        -------
        polaron_ZPR: real
            Zero-point renormalization, in eV
        polaron_inverse_lifetime: real
            Inverse lifetime of the polaron, in 1/ps
        effective_alpha: real
            Fröhlich effective alpha, dimensionless
        """
        if isinstance(band_mass, str):
            mass_info = np.loadtxt(band_mass, comments="#")
            weights = mass_info[:,0]
            qpoints_norm = mass_info[:,1:(1+self.num_dimensions)]
            masses = mass_info[:,(1+self.num_dimensions):]
            sqrt_band_masses = np.mean(np.sqrt(masses), axis=1)
        else:
            points, weights = scipy.integrate.lebedev_rule(lebedev_order)
            qpoints_norm = np.transpose(points)
            sqrt_band_masses = np.full_like(weights, np.sqrt(band_mass))
        weights_avg = weights/(4*np.pi)
        latvecs = self.get_lattice_vectors()
        small_radius = 0.01/np.sqrt(np.trace(latvecs @ np.transpose(latvecs)))
        q_points_direct = np.transpose(small_radius*
                                       (latvecs @ np.transpose(qpoints_norm)))
        omegas, eigvecs = \
            self.get_freqs_eigvecs_interpolated(q_points_direct, unit="eV", 
                convention="c-type", include_nac=include_nac, clean_value=1e-5)
        born_charges_3x3N = np.reshape(self.born_charges.T,
                                       (self.num_dimensions,self.numbands), 
                                       order='F')
        inv_mass_matrix = np.diag(1/np.sqrt(np.diag(self.get_mass_matrix())))
        mode_polarities_2 = np.abs(np.sum(np.tensordot(eigvecs, \
            np.transpose(born_charges_3x3N @ inv_mass_matrix), 1) \
            *qpoints_norm[:,None,:],2))**2
        dielectric_qs = np.sum((qpoints_norm @ self.dielectric_tensor)\
                               *qpoints_norm,1)
        alphas_qs_branches = 2.790068265963579*sqrt_band_masses[:,None]* \
            mode_polarities_2/ \
            (self.unitcell_volume*(omegas**2.5)*(dielectric_qs[:,None]**2))
        ## For future reference, the prefactor 2.79 is calculated as follows:
        # e = 1.602176634e-19  # Elementary charge, in Coulomb
        # hbar = 6.62607015e-34/(2*np.pi)  #Reduced Planck constant, in J.s
        # epsvac = 8.8541878188e-12  # Vacuum permittivity, in C^2/(J.m)
        # mel = 9.1093837139e-31  # Electron mass, in kg
        # amu = 1.66053906892e-27  # Atomic mass unit, in kg
        # prefactor = ((e**2/(4*np.pi*epsvac))**2)*4*np.pi*hbar*np.sqrt(mel)/\
        #     (np.sqrt(2)*1e-30*e**2.5*amu)  # equal to 2.79

        
        phonon_pops = n_BE(self.convert_units(omegas, from_unit="eV", 
                                              to_unit="THz"), temperature)
        # print(mode_polarities_2[0])
        # print(omegas[0]*1e3)
        # print(-weights_avg @ (alphas_qs_branches*omegas*(1+phonon_pops)))
        # print(self.convert_units(omegas[0], from_unit="eV", to_unit="THz"))
        polaron_ZPR = -weights_avg @ \
            np.sum(alphas_qs_branches*omegas*(1+phonon_pops),1)
        polaron_inverse_lifetime = (1/0.6582119569)*weights_avg @ \
            np.sum(alphas_qs_branches*omegas*phonon_pops,1)
        effective_alpha = weights_avg @ np.sum(alphas_qs_branches,1)
        return polaron_ZPR, polaron_inverse_lifetime, effective_alpha
        
        


    
    def plot_bands(self, path, path_labels, npoints=51, include_nac="None", 
                   unit="THz", title="Phonon dispersion", 
                   band_connections=False, band_label_permutations=None, 
                   highlight_degenerates=False, save_filename=None, 
                   text_sizes=(13, 15, 16), plot_range=None):
        """ Plot the phonon band structure in a figure

        Arguments
        ---------
        path: list of list of list of real
            High-symmetry path to be plotted on the Brillouin zone
            Follows the conventions of PhonoPy and pathsLabels.py: 
                - First level: list of connected path segments
                - Second level: list of points that mark path segments
                - Third level: direct coordinates of points
            Default: no path
        path_labels: list of str
            List of names of the edge points to be plotted on the path
            Default: no labels
        npoints: int
            Number of q-points on every path segment
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        unit: str
            Phonon frequency unit, default "THz"
        title: str
            Title of the plot
            Default: "Phonon dispersion"
        band_connections: bool
            Whether to calculate and plot the phonon band crossings
            Default: False
        band_label_permutations: np.array of int (only 0 or 1)
            shape (:, numbands, numbands)
            Label permutations output by self.get_label_permutations
            Only used when band_connections == True
            Default: None, recalculate from the eigenvectors
        highlight_degenerates: bool
            Whether to highlight the degenerate bands in red
            Default: False
        save_filename: str
            Save the figure with the given filename, in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        text_sizes: tuple of 3 ints
            Font sizes for small, medium and large text in the figure
                -Small text: axis ticks
                -Medium text: axis labels
                -Large text: title
            Default: (13, 15, 16)
        plot_range: tuple of 2 reals
            Minimum and maximum limits of the figure y-axis
            Default: rounded range of num_omegas points between the
            minimum and maximum phonon frequency
        
        Returns
        -------
        fig: figure handle
        ax: axis handle
        plot_handle: handle to the plot data

        """
        if band_connections:
            plot_style = dict(linestyle="solid", linewidth=1.5)
            plot_highlight_style = dict(color="black", linestyle="solid", 
                                        linewidth=2.0)
        else:
            plot_style = dict(color="black", linestyle="solid", linewidth=1.5)
            plot_highlight_style = dict(color="red", linestyle="solid", 
                                        linewidth=2.0)
        
        qs, distances, xaxis_labels, jump_indices = \
            self.parse_path(path, path_labels, npoints=npoints)
        
        omegas, eigenvectors = \
            self.get_freqs_eigvecs_interpolated(qs, unit=unit, 
                                                include_nac=include_nac)
        if band_connections:
            if band_label_permutations is None:
                band_label_permutations = \
                    self.get_label_permutations(qs, eigenvectors, 
                                                jump_indices=jump_indices)
        else:
            band_label_permutations = \
                np.tile(np.identity(self.numbands, dtype=int), 
                        (len(omegas),1,1))
        omegas_permuted = np.array([omegas[i] @ band_label_permutations[i] 
                                    for i in range(len(omegas))])

        
        fig, ax = plt.subplots()
        plot_handle = ax.plot(distances, omegas_permuted, **plot_style)
            
        if highlight_degenerates:
            # Plot the degenerate bands
            degenerates = np.full((len(omegas), self.numbands), False)
            max_omega = np.max(omegas)
            degenerates[:,1:] = \
                np.array([(omegas[:,i+1]-omegas[:,i])/max_omega < 2.0e-3 
                          for i in range(self.numbands-1)]).T
            omegas_degenerate = np.ma.masked_array(omegas, ~degenerates)
            ax.plot(distances, omegas_degenerate, **plot_highlight_style)
        
        xaxis_ticks = np.append(distances[0::npoints], distances[-1])
        
        ax.set_title(title, size=text_sizes[2])
        ax.set_ylabel('Phonon frequencies ('+str(unit)+')', 
                      fontsize=text_sizes[1])
        ax.set_xticks(xaxis_ticks, xaxis_labels)
        ax.tick_params(axis='both', labelsize=text_sizes[0])
        ax.set_xlim(np.min(distances),np.max(distances))
        
        if plot_range is None:
            omega_min = np.min(omegas)
            omega_max = np.max(omegas)
            if omega_min < -self.convert_units(0.1, from_unit="THz", 
                                               to_unit=unit):
                # Include the unstable phonon modes in the plot
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max)
            else:
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max, clamp_min=0)
        else:
            omega_min_scale = plot_range[0]
            omega_max_scale = plot_range[1]
        ax.set_ylim(omega_min_scale, omega_max_scale)
        
        # Plot major axis ticks
        for tick in xaxis_ticks:
            plt.axvline(x=tick, color='black', linewidth=1.0)
            
        # Plot unstable region in case of imaginary phonon frequencies
        if omega_min_scale < 0:
            plt.axhline(y=0.0, color='black', linewidth=1.0, linestyle="dashed")
            ax.add_patch(plt.Rectangle((distances[0], omega_min_scale), 
                                       distances[-1]-distances[0], 
                                       -omega_min_scale, fill=True, 
                                       color=(0.8, 0.8, 0.8), zorder=-10))

        fig.tight_layout()    
        fig.show()
        if save_filename is not None:
            create_path(save_filename)
            plt.savefig(save_filename+".pdf")
        return fig, ax, plot_handle
    
    def plot_LATO_weights(self, path, path_labels, npoints=51, 
                          include_nac="None", unit="THz", 
                          band_connections=False, band_label_permutations=None, 
                          highlight_degenerates=False, save_filename=None, 
                          text_sizes=(13, 15, 16), plot_range=None, 
                          num_markers=None, subplots=True, marker_style=None):
        """ Plot phonon band structure with LATO weights superimposed

        Arguments
        ---------
        path: list of list of list of real
            High-symmetry path to be plotted on the Brillouin zone
            Follows the conventions of PhonoPy and pathsLabels.py: 
                - First level: list of connected path segments
                - Second level: list of points that mark path segments
                - Third level: direct coordinates of points
            Default: no path
        path_labels: list of str
            List of names of the edge points to be plotted on the path
            Default: no labels
        npoints: int
            Number of q-points on every path segment
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        unit: str
            Phonon frequency unit, default "THz"
        band_connections: bool
            Whether to calculate and plot the phonon band crossings
            Default: False
        band_label_permutations: np.array of int (only 0 or 1)
            shape (:, numbands, numbands)
            Label permutations output by self.get_label_permutations
            Only used when band_connections == True
            Default: None, recalculate from the eigenvectors
        highlight_degenerates: bool
            Whether to highlight the degenerate bands in red
            Default: False
        save_filename: str
            Save the figure with the given filename, in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        text_sizes: tuple of 3 ints
            Font sizes for small, medium and large text in the figure
                -Small text: axis ticks
                -Medium text: axis labels
                -Large text: title
            Default: (13, 15, 16)
        plot_range: tuple of 2 reals
            Minimum and maximum limits of the figure y-axis
            Default: rounded range of num_omegas points between the
            minimum and maximum phonon frequency
        num_markers: int
            Number of markers plot on the bands
            Default: 10 markers per path segment
        subplots: bool
            Indicate whether to plot four subplots in one figure
            or whether to have each weight in its own figure
            Default: True, four subplots in one figure
        marker_style: dict
            Style parameters for the markers superimposed on the plot
            Default: dict(marker='o', color='lightblue', 
                edgecolors='black', linewidth=0.5, alpha=1.0)
            
        
        Returns
        -------
        fig_handles: list of fig_handle
            List of all the figure handles that were generated by
            this function

        """
        if band_connections:
            plot_style = dict(linestyle="solid", linewidth=1)
            plot_highlight_style = dict(color="black", linestyle="solid", 
                                        linewidth=1.5)
        else:
            plot_style = dict(color="black", linestyle="solid", linewidth=1)
            plot_highlight_style = dict(color="red", linestyle="solid", 
                                        linewidth=1.5)
        if num_markers is None:
            num_markers = sum([len(segment) for segment in path])*10 + 1
        if marker_style is None:
            marker_style = dict(marker='o', color='lightblue', 
                                edgecolors='black', linewidth=0.5, alpha=1.0)
        
        qs, distances, xaxis_labels, jump_indices = \
            self.parse_path(path, path_labels, npoints=npoints)
        omegas, eigenvectors = \
            self.get_freqs_eigvecs_interpolated(qs, unit=unit, 
                                                include_nac=include_nac)
        if band_connections:
            if band_label_permutations is None:
                band_label_permutations = \
                    self.get_label_permutations(qs, eigenvectors, 
                                                jump_indices=jump_indices)
        else:
            band_label_permutations = \
                np.tile(np.identity(self.numbands, dtype=int), 
                        (len(omegas),1,1))
        omegas_permuted = np.array([omegas[i] @ band_label_permutations[i]
                                    for i in range(len(omegas))])
        
        distances_markers = np.linspace(np.min(distances), np.max(distances), 
                                        num_markers)
        distances_array = np.tile(distances_markers, (self.numbands,1)).T
        qs_markers = scipy.interpolate.griddata(distances, qs, 
                                                distances_markers)
        
        omegas_markers, eigvecs_markers = \
            self.get_freqs_eigvecs_interpolated(qs_markers, unit=unit, 
                                                convention="c-type", 
                                                include_nac=include_nac, 
                                                clean_value=1e-3)
        weight_matrices = self.get_LATO_weights(qs_markers, 
                                                eigenvectors=eigvecs_markers)
        
        classes = ["TA", "LA", "TO", "LO"]
        marker_max_radius = 10
        if subplots is False:
            fig_handles = []
            ax_handles = []
            save_filenames = []
            figsize = (6.4, 4.8)
            for LATO_class in classes:
                fig, ax = plt.subplots(figsize=figsize)
                fig_handles.append(fig)
                ax_handles.append(ax)
                if save_filename is not None:
                    save_name = save_filename+"_"+LATO_class
                else:
                    save_name = None
                save_filenames.append(save_name)              
        else:
            figsize = (6.4*2, 4.8*2)
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            fig_handles = [fig]
            ax_handles = axs.flat
            save_filenames = [save_filename]
        plot_handles = []
        for index, ax in enumerate(ax_handles):
            marker_sizes = np.real(marker_max_radius**2*\
                                   weight_matrices[:,index,:])            
            plot_handles_this = ax.plot(distances, omegas_permuted, 
                                        **plot_style, zorder=5)
            if highlight_degenerates:
                # Plot the degenerate bands
                degenerates = np.full((len(omegas), self.numbands), False)
                max_omega = np.max(omegas)
                degenerates[:,1:] = \
                    np.array([(omegas[:,i+1]-omegas[:,i])/max_omega < 2.0e-3 
                              for i in range(self.numbands-1)]).T
                omegas_degenerate = np.ma.masked_array(omegas_permuted, 
                                                       ~degenerates)
                ax.plot(distances, omegas_degenerate, 
                        **plot_highlight_style, zorder=6)
            
            plot_handle_markers = ax.scatter(distances_array, omegas_markers, 
                                             s=marker_sizes, **marker_style, 
                                             zorder=0)
            plot_handles_this.append(plot_handle_markers)
            
            xaxis_ticks = np.append(distances[0::npoints], distances[-1])
            
            ax.set_title(classes[index]+" weight", size=text_sizes[2])
            ax.set_ylabel('Phonon frequencies (THz)', fontsize=text_sizes[1])
            ax.set_xticks(xaxis_ticks, xaxis_labels)
            ax.tick_params(axis='both', labelsize=text_sizes[0])
            ax.set_xlim(np.min(distances),np.max(distances))
            
            if plot_range is None:
                omega_min = np.min(omegas)
                omega_max = np.max(omegas)
                if omega_min < -self.convert_units(0.1, from_unit="THz", 
                                                   to_unit=unit):
                    # Include the unstable phonon modes in the plot
                    omega_min_scale, omega_max_scale = \
                        round_plot_range(omega_min, omega_max)
                else:
                    omega_min_scale, omega_max_scale = \
                        round_plot_range(omega_min, omega_max, clamp_min=0)
            else:
                omega_min_scale = plot_range[0]
                omega_max_scale = plot_range[1]
            ax.set_ylim(omega_min_scale, omega_max_scale)
            
            # Plot major axis ticks
            for tick in xaxis_ticks:
                ax.axvline(x=tick, color='black', linewidth=0.5)
            
            # Plot unstable region in case of imaginary phonon frequencies
            if omega_min_scale < 0:
                ax.axhline(y=0.0, color='black', linewidth=1.0, 
                           linestyle="dashed")
                ax.add_patch(plt.Rectangle((distances[0], omega_min_scale), 
                                           distances[-1]-distances[0], 
                                           -omega_min_scale, fill=True, 
                                           color=(0.8, 0.8, 0.8), zorder=-10))
                
            plot_handles.append(plot_handles_this)
            if save_filename is not None and subplots is False:
                create_path(save_filename)
                fig_handles[index].tight_layout()
                fig_handles[index].savefig(save_filename+"_"+\
                                           classes[index]+".pdf")
        for fig, save_name in zip(fig_handles, save_filenames):
            fig.tight_layout()
            fig.show()
            if save_name is not None:
                create_path(save_name)
                fig.savefig(save_name+".pdf")
        return fig_handles

class YCalculation():
    """ Collection of calculations at specified electric fields

    This class has a list of PhonopyCommensurateCalculations as
    attributes, where each of these calculations is supposed to be
    calculated at different external electric fields.
    These calculations are used to calculate the electric field
    derivative of the dynamical matrix, and its derived quantities
    such as Y_{nu1,nu2}(q) and T(omega).
    The code assumes that one of the calculations is performed
    at zero electric field, and that all calculations are performed
    on the same grid of commensurate q-points.
    
    Attributes
    ----------
    calcs: list of PhonopyCommensurateCalculation
    Efields: np.array of real
        shape (len(calcs),)
    num_calcs: int
        Equal to len(calcs)
    zerocalc: PhonopyCommensurateCalculation
        The calculation that is performed at E=0
    dynmats_derE: np.array of complex
        shape (:, numbands, numbands)
    take_imag: bool

    """
    def __init__(self, yaml_filenames, Efields, born_filename=None, 
                 Efield_unit='V/Ang', take_imag=False, 
                 nac_q_direction=np.array([1.0,0.0,0.0]), nac_G_cutoff=None, 
                 nac_lambda=None):
        """ YCalculation(yaml_filenames, Efields, ...)

        Arguments
        ---------
        yaml_filenames: list of str
            List of the .yaml files that contain the PhonoPy
            calculation data to load in
        Efields: list of real
            List of electric fields that the PhonoPy calculations
            were performed at
        born_filename: str
            BORN file that contains the Born effective charge tensors
            and dielectric tensors.
            Important: this function expects a BORN file with a line
            for each atom, since no symmetry is implemented. PhonoPy
            exports a BORN file with less lines based on symmetry,
            which is not compatible with this code.
        Efield_unit: str
            Units in which the electric field are given
            Default: "V/Ang", default units in VASP
        take_image: bool
            Whether to take the imaginary part of the dynamical matrix
            derivative. More accurate results for materials with an 
            inversion center that satisfies R(k)=k, but wrong results
            for other materials
            Default: False
        nac_q_direction: np.array of real
            In case the dynamical matrix is not analytic at Gamma,
            store the values with limiting direction nac_q_direction
            Default: np.array([1.0, 0.0, 0.0])
        nac_G_cutoff: real
            Cutoff radius for reciprocal lattice sums
            Default: Same as PhonoPy, chosen to include approximately 
            300 reciprocal lattice points
        nac_lambda: real
            Exponential decay for reciprocal lattice sums
            Default: Same as PhonoPy, based on self.dielectric_tensor
            and self.nac_G_cutoff

        Raises
        ------
        ValueError
            when the given number of Efields does not match the
            number of given calculations

        """
        self.num_calcs = len(yaml_filenames)
        if len(Efields) != self.num_calcs:
            raise ValueError("""The given number of Efields does not
                              match the number of given calculations""")
        self.Efields = \
            self.convert_Efield_units(np.array(Efields), 
                                      from_unit=Efield_unit, to_unit='V/Ang')
        self.calcs = [PhonopyCommensurateCalculation(filename, born_filename, 
                                                     nac_q_direction, 
                                                     nac_G_cutoff, nac_lambda)
                      for filename in yaml_filenames]
        self.zerocalc = self.calcs[np.argmin(np.abs(self.Efields))]
        self.set_take_imag(take_imag)
        self.dynmats_derE = \
            self.get_dynmats_derE(freq_unit='THz', convention="c-type", 
                                  Efield_unit='V/Ang', take_imag=self.take_imag)
        
    def set_take_imag(self, take_imag):
        """ Set take_imag to a certain value """
        if take_imag is not None:
            self.take_imag = take_imag
        
    def get_num_calcs(self):
        """ Get the number of calculations """
        return self.num_calcs
    
    def get_zerocalc(self):
        """ Get the calculation at zero electric field """
        return self.zerocalc
        
    def get_calcs(self):
        """ Get a list of all calculations """
        return self.calcs
    
    def get_Efields(self, unit='V/Ang'):
        """ Get a list of the electric fields in the desired unit """
        return self.convert_Efield_units(self.Efields, from_unit='V/Ang', 
                                         to_unit=unit)
    
    def convert_Efield_units(self, Efields, from_unit='V/Ang', to_unit='V/Ang'):
        """ Convert between different units for the electric field

        The supported units are:
            - "V/Ang": Volts per angstroms, also sometimes written as
                       the force eV/Ang
            - "V/m": Volts per meter, SI base unit equivalent to N/C
            - "esu": statV/cm, as in Gaussian or ESU units

        Arguments
        ---------
        Efields: real or np.array of real
            Electric field given in from_unit 
        from_unit: str
            Unit that Efields is written in
        to_unit: str
            Desired output unit

        Returns
        -------
        Efields in units given by to_unit
        """

        match from_unit:
            case "V/Ang":
                Efield_in_Vang = Efields
            case "V/m":
                Efield_in_Vang = Efields*1e-10
            case "esu":
                Efield_in_Vang = Efields*1e-16*299792458
            case _:
                warn_string = str(from_unit)+\
                """ is not a recognized electric field unit.
                Currently only 'V/ang', 'V/m', and 'esu' are supported.
                It is assumed that the input electric fields were in V/Ang."""
                warnings.warn(warn_string)
                Efield_in_Vang = Efields
        match to_unit:
            case "V/Ang":
                return Efield_in_Vang
            case "V/m":
                return Efield_in_Vang*1e10
            case "esu":
                return Efield_in_Vang*1e16/299792458
            case _:
                warn_string = str(to_unit)+\
                """ is not a recognized phonon frequency unit.
                Currently only 'V/ang', 'V/m' and 'esu' are supported.
                Electric fields in V/Ang were returned instead."""
                warnings.warn(warn_string)
                return Efield_in_Vang
    
    def get_dynmats_derE(self, freq_unit='THz', Efield_unit='V/Ang', 
                         convention="c-type", take_imag=None):
        """ Get dynamical matrix derivative from finite differences

        Uses an n-th order finite difference approximation to calculate
        the derivative of the dynamical matrix with respect to an
        electric field.
        The dynamical matrix data is automatically obtained from
        the calculations contained in self.calcs, and the electric
        fields are obtained from self.get_Efields

        If take_imag is False, at least two electric field calculations
        are necessary to approximatethe derivative.

        Arguments
        ---------
        freq_unit: str
            Phonon frequency unit, default "THz"
        Efield_unit: str
            Electric field unit, default "V/Ang"
        convention: str
            Convention for the dynamical matrix, default "c-type"
        take_imag: bool
            Whether to take the imaginary part of the result. Also sets
            self.take_imag to the given value. Only correct when the
            material has an inversion center that satisfies R(k)=k. 
            Default: self.take_imag

        Returns
        -------
        dynmat_der: np.array of complex
            shape (numqpoints, numbands, numbands)
            Derivative of the dynamical matrix with respect to an
            external electric field

        Raises
        ------
        ValueError
            when take_imag is False and only one electric field 
            calculation is available

        """
        self.set_take_imag(take_imag)
        Efields = self.get_Efields(unit=Efield_unit)
        if len(Efields) == 1:
            # Calculating the electric field derivative is only possible 
            # by taking the imaginary part
            if self.take_imag:
                weights = np.ones(1) / Efields[0]
            else:
                raise ValueError(
                    """Calculating the derivative with take_imag=False
                    requires at least two electric fields."""
                    )
        else:
            # Make a Vandermonde matrix of the electric fields, 
            # and solve it to get the finite difference coefficients:
            Efield_scale = np.max(np.abs(Efields))
            xs = Efields / Efield_scale
            system_matrix = np.vander(xs, increasing=True).T
            ordinates = np.zeros(len(xs))
            ordinates[1] = 1
            weights = np.linalg.solve(system_matrix, ordinates) / Efield_scale
        dynmats_at_Es = \
            np.array([calc.get_dynamical_matrices(unit=freq_unit, 
                                                  convention=convention) 
                      for calc in self.calcs])
        if self.take_imag:
            return 1j*np.imag(np.moveaxis(dynmats_at_Es, 0, -1) @ weights)
        else:
            return np.moveaxis(dynmats_at_Es, 0, -1) @ weights
        
    def dynmats_derE_interpolate(self, q, freq_unit='THz', Efield_unit='V/Ang', 
                                 convention="c-type"):
        """ Interpolate the dynamical matrix derivative to any q-point

        Arguments
        ---------
        q: np.array of real
            shape (:, 3)
            Array of q-points at which the dynamical matrix derivative
            is to be calculated
        freq_unit: str
            Phonon frequency unit, default "THz"
        Efield_unit: str
            Electric field unit, default "V/Ang"
        convention: str
            Convention for the dynamical matrix, default "c-type"

        Returns
        -------
        dynmat_der: np.array of complex
            shape (len(q), numbands, numbands)
            Interpolated derivative of the dynamical matrix
            with respect to an external electric field

        """
        quantity = self.get_dynmats_derE(freq_unit=freq_unit, 
                                         convention=convention, 
                                         Efield_unit=Efield_unit)
        return self.zerocalc.fourier_interpolate(q, quantity=quantity, 
                                                 unit=freq_unit, 
                                                 convention=convention, 
                                                 include_nac="None")
        
        
    def Y_interpolate(self, q, include_nac="None", freqs=None, eigvecs=None,
                      no_freqs=False, imaginary_cutoff=0.001):
        """ Calculate Y_{\nu_1,\nu_2}(q) at arbitrary q-points

        Note that include_nac only refers to the harmonic phonon
        quantities, there is no implementation yet for the
        1-electron-2-phonon NAC contribution

        Arguments
        ---------
        q: np.array of real
            shape (:, 3)
            Array of q-points at which the dynamical matrix derivative
            is to be calculated
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
        freqs: np.array of real
            shape (:, numbands)
            Phonon frequencies evaluated at input q
            Default: recalculated from self.zerocalc

        eigvecs: np.array of complex
            shape (:, numbands, numbands)
            Phonon eigenvectors evaluated at input q
            Default: recalculated from self.zerocalc

        no_freqs: bool
            If True, do not divide by the square roots of phonon
            frequencies, or any prefactors. The returned Y is given
            in the same units as dD/dE, which is THz²*Å/V.
            Default: False

        imaginary_cutoff: real
            Any phonon branches with frequencies (in THz) lower than
            imaginary_cutoff will have their value of Y set to zero.
            Can be used to remove unstable phonon branches from
            the calculations in a first approximation.
            Default: 0.001

        Returns
        -------
        Y: np.array of complex
            shape (len(q), numbands, numbands)
            Represents Y_{\nu_1,\nu_2}(q), the strengths of the
            1-electron-2-phonon processes, expressed in 1e-10 m

        """

        outer_product = lambda A: A[...,:,None]*A[...,None,:]
        if freqs is None or eigvecs is None:
            # We divide by square roots of the phonon frequencies, so they
            # must be positive: therefore we set clean_value
            freqs, eigvecs = self.zerocalc.get_freqs_eigvecs_interpolated(\
                q, unit="THz", convention="c-type", include_nac=include_nac, 
                clean_value=imaginary_cutoff)
        mask = outer_product(freqs > imaginary_cutoff)
        
        dynmats_derE = self.dynmats_derE_interpolate(q, freq_unit="THz", 
                                                     convention="c-type", 
                                                     Efield_unit='V/Ang')
        if no_freqs:
            return -1j * mask * \
                (eigvecs.conj() @ dynmats_derE @ np.swapaxes(eigvecs,-1,-2))
        else:
            freqs_invsqrt = outer_product(1/np.sqrt(freqs))
            return -0.5j * 0.004135667 * freqs_invsqrt* mask * \
                (eigvecs.conj() @ dynmats_derE @ np.swapaxes(eigvecs,-1,-2))
    
    def calculate_Tomega(self, q_mesh_size=8, unit="THz", include_nac="None", 
                         sigma=None, omega=None, num_omegas=1001, 
                         imaginary_cutoff=0.001,
                         moments=np.array([-0.5, -1.0, -1.5]), 
                         moments_scaling_frequency=None, temperature=0.,
                         q_split_levels=0, parallel_jobs=1, 
                         savedata_filename=None):
        """ Calculate and plot T(omega) with the smearing method

        This function offers functionality to calculate the 
        LATO-resolved 1-electron-2-phonon spectral function,
        write the results to a file or save them in a .npz file,
        and plotting the result in a graph.
        The Brillouin zone integral is calculated with the trapezoid
        rule, with Gaussian smearing for the delta functions

        Arguments
        ---------
        q_mesh_size: int, or np.array of int
            Determines the size of the N x N x N fine q-grid which is
            used for Brillouin zone integration
            Default: 8
        unit: str
            Phonon frequency unit, default "THz"
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        sigma: real
            Gaussian smearing width for the delta functions, in units
            of the phonon frequency
            It's safer to always specify a value explicitly
            Default: 0.01*(2*max(omega_ph) - min(omega_ph))
        omega: np.array of real
            shape (:, )
            Frequencies at which T(omega) is to be calculated
            Default: num_omega points between 0 and 2*max(omega_ph),
            rounded up
        num_omegas: int
            Number of frequencies if no array is specified for omega
            Default: 1001
        imaginary_cutoff: real
            Any phonon branches with frequencies (in THz) lower than
            imaginary_cutoff will have their value of Y set to zero.
            Can be used to remove unstable phonon branches from
            the calculations in a first approximation.
            Default: 0.001
        moments: np.array of real
            shape (:, )
            Exponents for the moments of T(omega) to be calculated
            Default: np.array([-0.5, -1.0, -1.5])
        moments_scaling_frequency: real
            Scaling frequency to make the moments dimensionless
            Stored values are T_n/(moments_scaling_frequency)^(n+1)
            Default: highest phonon frequency at Gamma
        temperature: real
            Temperature at which to calculate Tomega, in Kelvin
            Default: 0 K
        q_split_levels: int, between 0 and 3
            Controls how many of the fine q-mesh points are generated
            at once. In general, only one (3 - q_split_levels)-
            dimensional slice of the q-mesh is generated and stored 
            at a time. The calculation is parallellized over each
            slive. Useful for very large q-meshes which would
            otherwise take up too much memory (e.g. 128x128x128).
            Default: 0, the entire array is generated at once
        parallel_jobs: int
            Number of parallel processes used for the Brillouin zone
            integral, can only be used when q_split_levels > 0
            Default: 1
        savedata_filename: str
            Filename to store the resulting data in an .npz file
            The following variables will be stored:
                - q_mesh_size
                - unit
                - include_nac
                - sigma
                - moments
                - moments_scaling_frequency
                - omega
                - omega_max, upper end of the frequency range
                - Tomega, 1-electron-2-phonon spectral function
                - Tomega_AO, with acoustic/optical distinction
                - Tomega_LT, with longitudinal/transverse distinction
                - Tomega_LATO, with full LATO distinction
                - Tmoments, moments of the 1e2ph spectral function
                - Tmoments_LT, with acoustic/optical distinction
                - Tmoments_AO, with longitudinal/transverse distinction
                - Tmoments_LATO, with full LATO distinction
            Default: None, the calculation data is not saved

        Returns
        -------
        omega: np.array of real
            shape (:, )
            Frequencies at which T(omega) is calculated, in input units
        Tomega: np.array of real
            shape (len(omega), 4, 4)
            LATO-resolved T(omega) at the given frequencies
        Tmoments: np.array of real
            shape (len(moments), 4, 4)
            LATO-resolved moments of T(omega)
        fig: figure handle
        ax: axis handle
        plot_handle: handle to the plot data

        Raises
        ------
        ValueError
            when q_mesh_size is not an int
            or an np.array of int of shape (3,)
        ValueError
            when q_split_levels is not between 0 and 3

        """
        

        phonon_freqs_input_units = self.zerocalc.get_frequencies(unit)
        omega_max = np.max(phonon_freqs_input_units)*2
        omega_min = np.min(phonon_freqs_input_units)
        if moments_scaling_frequency is None:
            # Choose the highest frequency at Gamma as the 
            # moments scaling frequency by default
            moments_scaling_frequency = \
                np.max(self.zerocalc.get_freqs_eigvecs_interpolated(\
                    np.array([0.0,0.0,0.0]), include_nac="Gonze", unit=unit)[0])
        if omega is None:
            # Set the default range of omega if none is provided
            rounding_scale = 10**np.floor(np.log10(abs(omega_max)))
            omega_min_scale = np.trunc(omega_min/rounding_scale)*rounding_scale 
            omega_max_scale = np.ceil(omega_max/rounding_scale)*rounding_scale 
            omega_input_units = np.linspace(omega_min_scale, omega_max_scale, 
                                            num=num_omegas, endpoint=True)
        else:
            omega_input_units = omega
        omega = self.zerocalc.convert_units(omega_input_units, from_unit=unit, 
                                            to_unit="THz")
        if not sigma:
            # Set the default value of sigma if none is provided
            sigma = 0.01*(omega_max - omega_min)
        else:
            # We must convert sigma from the input units to THz
            sigma = self.zerocalc.convert_units(sigma, from_unit=unit, 
                                                to_unit="THz")
        
        # Convert input q_mesh_size_array into the correct format,
        # a 1D array of size num_dimensions:
        match q_mesh_size:
            case int():
                q_mesh_size_array = np.repeat(q_mesh_size, 
                                              self.zerocalc.num_dimensions)
            case np.ndarray():
                q_mesh_size_array = np.floor(q_mesh_size).astype(int).flatten()\
                    [0:self.zerocalc.num_dimensions]
            case _:
                raise ValueError("""q_mesh_size must be one integer
                                 or a numpy array of integers""")
        q_split_levels = int(q_split_levels)
        if q_split_levels < 0 or q_split_levels > self.zerocalc.num_dimensions:
            raise ValueError("""q_split_levels must be between 0 and 
                             the number of dimensions (usually 3)""")
        if q_split_levels == 0:
            # We store all q-points in one big array and pass them all at once
            qs = self.zerocalc.get_dense_qmesh(q_mesh_size_array)
            Tomega_LATO = self.Tomega_LATO_smearing(omega, qs, sigma,
                                    temperature, include_nac=include_nac)
        else:
            # We generate the q-points in smaller arrays, with q_split_levels 
            # indices fixed, and then add everything up.
            # This is slower but uses much less memory, since we don't need to 
            # store a massive q-point array.
            # Use this when the requested q_mesh_size is rather large, 
            # depending on your system
            qs_fixed = self.zerocalc.get_dense_qmesh(\
                q_mesh_size_array[-q_split_levels:], 
                num_dimensions=q_split_levels)
            if parallel_jobs > 1:
                T_omega_terms = joblib.Parallel(n_jobs=parallel_jobs)(\
                    joblib.delayed(self.Tomega_LATO_smearing)(\
                        omega, self.zerocalc.get_dense_qmesh(\
                            q_mesh_size_array[:-q_split_levels], 
                            fixed_indices=q_fixed), 
                        sigma, temperature, include_nac=include_nac,
                        imaginary_cutoff=imaginary_cutoff) 
                    for q_fixed in qs_fixed)
            else:
                T_omega_terms = [self.Tomega_LATO_smearing(omega, 
                        self.zerocalc.get_dense_qmesh(\
                            q_mesh_size_array[:-q_split_levels], 
                            fixed_indices=q_fixed), 
                        sigma, temperature, include_nac=include_nac,
                        imaginary_cutoff=imaginary_cutoff) 
                    for q_fixed in qs_fixed]
            Tomega_LATO = sum(T_omega_terms) / len(qs_fixed)
        
        # Get all the different resolutions from Tomega_LATO
        partials_list = [[[0],[1],[2],[3]], [[0,2],[1,3]],
                         [[0,1],[2,3]], [[0,1,2,3]]]
        # In order, this is LATO resolution, LT resolution, AO resolution, 
        # and no resolution
        Tomega_resolutions = []
        Tmoments_resolutions = []
        for partials in partials_list:
            Tomega_current = np.empty((len(omega_input_units), len(partials), 
                                       len(partials))) 
            for index2, sum_indices_x in enumerate(partials):
                for index3, sum_indices_y in enumerate(partials):
                    sum_where = np.zeros_like(Tomega_LATO[0], dtype=bool)
                    sum_where[np.ix_(sum_indices_x, sum_indices_y)] = True
                    Tomega_current[:, index2, index3] = \
                        np.sum(Tomega_LATO, axis=(1,2), where=sum_where)
            Tmoments_current = np.zeros((len(moments), len(partials), 
                                            len(partials)))
            nonzero_indices = np.abs(omega_input_units)>1e-10
            nonzero_omegas = omega_input_units[nonzero_indices]
            for index, n in enumerate(moments):
                omega_power = np.zeros_like(omega_input_units)
                omega_power[nonzero_indices] = np.power(nonzero_omegas, n)
                Tmoments_current[index, ...] = \
                    np.power(moments_scaling_frequency, -(n+1)) * \
                    scipy.integrate.simpson(\
                        Tomega_current*omega_power[:,None,None],
                        x=omega_input_units, axis=0)
            Tomega_resolutions.append(Tomega_current)
            Tmoments_resolutions.append(Tmoments_current)
        # Name the resolutions and save them if necessary
        Tmoments_LATO = Tmoments_resolutions[0]
        Tomega_LT = Tomega_resolutions[1]
        Tmoments_LT = Tmoments_resolutions[1]
        Tomega_AO = Tomega_resolutions[2]
        Tmoments_AO = Tmoments_resolutions[2]
        Tomega = Tomega_resolutions[3][:,0,0]
        Tmoments = Tmoments_resolutions[3][:,0,0]

        results_dict = {
            "q_mesh_size": q_mesh_size,
            "unit": unit,
            "include_nac": include_nac,
            "sigma": sigma,
            "moments": moments,
            "moments_scaling_frequency": moments_scaling_frequency,
            "omega": omega_input_units,
            "omega_max": omega_max,
            "Tomega_LATO": Tomega_LATO,
            "Tomega_LT": Tomega_LT,
            "Tomega_AO": Tomega_AO,
            "Tomega": Tomega,
            "Tmoments_LATO": Tmoments_LATO,
            "Tmoments_LT": Tmoments_LT,
            "Tmoments_AO": Tmoments_AO,
            "Tmoments": Tmoments,
            "temperature": temperature
        }
        results = TomegaResults(results_dict)
        if savedata_filename is not None:
            results.save_npz(savedata_filename)

        return results
    
    def Tomega_LATO_smearing(self, omegas, qs, sigma, temperature,
                             include_nac="None", imaginary_cutoff=0.001):
        """ Compute LATO-resolved T(omega) with the smearing method
        
        Uses the trapezoid rule for Brillouin zone integration

        This function isn't supposed to be called by the user,
        it simply summarizes the used algorithm into a single function

        Arguments
        ---------
        omegas: np.array of real
            shape (:)
            Array of frequencies at which T(omega) is to be evaluated
        qs: np.array of real
            shape (:, 3)
            Dense mesh of q-points for the Brillouin zone integration
        sigma: real
            Delta function smearing width, in THz
        temperature: real
            Temperature, in Kelvin
        include_nac: str
            Mode of NAC correction, "None", "Gonze", or "Wang"
        imaginary_cutoff: real
            Any phonon branches with frequencies (in THz) lower than
            imaginary_cutoff will have their value of Y set to zero.
            Can be used to remove unstable phonon branches from
            the calculations in a first approximation.
            Default: 0.001

        Returns
        -------
        T_omega: np.array of real
            shape (len(omegas), 4, 4)
            LATO-resolved T(omega) evaluated at the input omegas

        """
        numbands = self.zerocalc.numbands
        # Prefactor for T(omega), omega in THz:
        eV_to_dimensionless = 21876.91262642702/self.zerocalc.unitcell_volume
        # Define the smeared delta function:
        two_sigma_squared = 2*sigma*sigma
        delta_smeared = lambda x : np.exp(- x*x/two_sigma_squared)\
            /np.sqrt(np.pi*two_sigma_squared)
        # Calculate the interpolated phonon frequencies
        # Note: we clean the acoustic frequencies because we divide by them
        phonon_freqs, eigvecs = self.zerocalc.get_freqs_eigvecs_interpolated(\
            qs, unit="THz", include_nac=include_nac, convention="c-type",
            clean_value=imaginary_cutoff)
        # Calculate the LATO weights for each band:
        LATO_weights = self.zerocalc.get_LATO_weights(qs, eigenvectors=eigvecs)
        # Define array that represents n(omega_{q, nu})
        phonon_occs = n_BE(phonon_freqs, temperature)
        # Define array that represents omega_{q, nu1} + omega_{q, nu2}
        omega_sum = phonon_freqs.reshape((-1,numbands,1))\
              + phonon_freqs.reshape((-1,1,numbands))
        # Define array that represents 1 + n(omega_{q, nu1}) + n(omega_{q, nu2})
        omega_nB_sum = 1. + phonon_occs.reshape((-1,numbands,1))\
              + phonon_occs.reshape((-1,1,numbands))
        # Define array that represents |omega_{q, nu1} - omega_{q, nu2}|
        omega_diff = np.abs(phonon_freqs.reshape((-1,numbands,1))\
              - phonon_freqs.reshape((-1,1,numbands)))
        # Define array that represents |n(omega_{q, nu1}) - n(omega_{q, nu2})|
        omega_nB_diff = np.abs(phonon_occs.reshape((-1,numbands,1))\
              - phonon_occs.reshape((-1,1,numbands)))
        # Define array that represents |Y_{nu1, nu2}(q)|^2
        Y2s = np.abs(self.Y_interpolate(qs, include_nac=include_nac, 
                                        freqs=phonon_freqs, eigvecs=eigvecs,
                                        imaginary_cutoff=imaginary_cutoff))**2
        # Define T(omega) as calculated with the trapezoid rule:
        T_func = lambda x : eV_to_dimensionless * \
            np.mean(np.conjugate(LATO_weights)@(Y2s*(
                omega_nB_sum*delta_smeared(x-omega_sum) \
                + omega_nB_diff*delta_smeared(x-omega_diff)
                ))\
                     @ np.swapaxes(LATO_weights, -1, -2), axis=0)
        #Iterate T_func over all elements in omega:
        T_omega = np.empty((len(omegas), LATO_weights.shape[1], 
                            LATO_weights.shape[1]))
        for index, freq in enumerate(omegas):
            T_omega[index, ...] = np.real(T_func(freq)) # Ensure output is real
        return T_omega
    
    def plotY(self, path, path_labels, npoints=51, num_markers=None, 
              include_nac="None", title1=None, title2=None, 
              degenerate_cutoff=1e-3, Y2_norm_value=None, 
              Y2_sum_norm_value=None, band_label_permutations=None,
              save_filename=None, plot_style=None, plot_highlight_style=None, 
              marker_style=None, subplots=None, shareaxes=False, figsize=None, 
              text_sizes=(13, 15, 16), plot_range=None, no_freqs=False,
              imaginary_cutoff=0.001):
        
        """ Plot |Y_{nu_1,nu_2}(q)|^2 over the phonon band structure

        Makes an amount of plots of |Y_{nu_1,nu_2}(q)|^2 equal to
        numbands, each where one phonon band is kept constant. The
        markers are normalized to the maximum value of 
        |Y_{nu_1,nu_2}(q)|^2 over all the pairs of bands.
        Also makes a plot of sum_{nu_2}|Y_{nu_1,nu_2}(q)|^2.

        Arguments
        ---------
        path: list of list of list of real
            High-symmetry path to be plotted on the Brillouin zone
            Follows the conventions of PhonoPy and pathsLabels.py: 
                - First level: list of connected path segments
                - Second level: list of points that mark path segments
                - Third level: direct coordinates of points
            Default: no path
        path_labels: list of str
            List of names of the edge points to be plotted on the path
            Default: no labels
        npoints: int
            Number of q-points on every path segment
            Default: 51
        num_markers: int
            Number of markers to plot over the length of the path
            Default: 10 for every path segment
        include_nac: str
            Indicate what kind of non-analytic correction to include:
                - "None": no non-analytic correction, gives unphysical 
                  results for polar materials
                - "Gonze": PhonoPy default method, requires BORN input
                  X. Gonze and C. Lee, Phys. Rev. B 55, 10355 (1997)
                - "Wang": Method based on Y Wang et al., 
                  J. Phys.: Condens. Matter 22 202201 (2010)
            Default: "None"
        unit: str
            Phonon frequency unit, default "THz"
        title1: str
            Title of the numbands subplots
            Default: "$|Y_{\\nu i,z}(\\mathbf{q})|^2$", where i
            goes from 0 to numbands-1
        title2: str
            Title of the summarizing plot
            Default:"$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$"
        degenerate_cutoff: real
            Tolerance for determining when two bands are degenerate
            Default: 1e-3
        Y2_norm_value: real
            Value of |Y_{nu_1,nu_2}(q)|^2 that corresponds to the 
            marker of size max_radius (defined in marker_style)
            Default: maximum value of |Y_{nu_1,nu_2}(q)|^2
        Y2_sum_norm_value: real
            Value of sum_{nu_2} |Y_{nu_1,nu_2}(q)|^2 that 
            corresponds to the marker of size max_radius
            Default: equal to Y2_norm_value, markers have equal scale
        band_label_permutations: np.array of int (only 0 or 1)
            shape (:, numbands, numbands)
            Label permutations output by self.get_label_permutations
            Default: None, recalculate from the eigenvectors
        save_filename: str
            Save all figures with the given common filename, 
            in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        plot_style: dict
            Style parameters for the phonon band lines
            Default: dict(color="black",linestyle="solid",linewidth=1)
        plot_highlight_style: dict
            Style parameters for the highlighted phonon band
            Default: dict(color="red", linestyle="solid", linewidth=1)
        marker_style: dict
            Style parameters for the markers superimposed on the plot
            Default: dict(marker='o', color='lightblue', max_radius=10,
                          edgecolors='black', linewidth=0.5, alpha=1.0)
        subplots: tuple of 2 ints, or None
            If None, make numbands different figures
            Otherwise, make one figure with numbands subplots,
            arranged in an array of shape given by subplots
            Default: None, all different figures
        shareaxes: bool
            Sets the shareaxes property of the subplots, if subplots
            is not equal to None
            Default: False
        figsize: tuple of 2 ints
            Sets the size of the figures in inches
            Default: (6.4*subplots[1], 4.8*subplots[0])
        text_sizes: tuple of 3 ints
            Font sizes for small, medium and large text in the figure
                -Small text: axis ticks
                -Medium text: axis labels
                -Large text: title
            Default: (13, 15, 16)
        plot_range: tuple of 2 reals
            Minimum and maximum limits of the figure y-axis
            Default: rounded range of num_omegas points between the
            minimum and maximum phonon frequency
        no_freqs: bool
            If True, do not divide by the square roots of phonon
            frequencies, or any prefactors.
            Default: False
        imaginary_cutoff: real
            Any phonon branches with frequencies (in THz) lower than
            imaginary_cutoff will have their value of Y set to zero.
            Can be used to remove unstable phonon branches from
            the calculations in a first approximation.
            Default: 0.001
        
        Returns
        -------
        Y2_max_value: real
            Value of |Y_{nu_1,nu_2}(q)|^2 corresponding to the 
            largest marker 
        Y2_sum_max_value: real
            Value of sum_{nu_2} |Y_{nu_1,nu_2}(q)|^2 corresponding
            to the largest marker
        fig: figure handle
        ax: axis handle
        plot_handle: handle to the plot data

        """


        if plot_style is None:
            plot_style = dict(color="black", linestyle="solid", linewidth=1)
        if plot_highlight_style is None:
            plot_highlight_style = dict(color="red", linestyle="solid", 
                                        linewidth=1)
        if marker_style is None:
            marker_style = dict(marker='o', color='lightblue', 
                                edgecolors='black', linewidth=0.5, alpha=1.0,
                                max_radius=10)
        if figsize is None and subplots is not None:
            figsize = (6.4*subplots[1], 4.8*subplots[0])
        if num_markers is None:
            num_markers = sum([len(segment) for segment in path])*10 + 1
        marker_max_radius = marker_style['max_radius']
        del marker_style['max_radius']
        number_of_bands = self.zerocalc.numbands
        set_title1 = title1 is None
        set_title2 = title2 is None
        
        unit="THz"
        qs, distances, xaxis_labels, jump_indices = \
            self.zerocalc.parse_path(path, path_labels, npoints=npoints)
        omegas, eigvecs = self.zerocalc.get_freqs_eigvecs_interpolated(\
            qs, unit=unit, convention="c-type", include_nac=include_nac)
        omegas_calc = self.zerocalc.get_clean_frequencies(
            frequencies_to_clean = omegas, min_value=imaginary_cutoff)
        if band_label_permutations is None:
            band_label_permutations = self.zerocalc.get_label_permutations(\
                qs, eigvecs, jump_indices)
            
        distances_markers = np.linspace(np.min(distances), np.max(distances), 
                                        num_markers)
        distances_array = np.tile(distances_markers, (number_of_bands,1)).T
        # qs_markers = scipy.interpolate.griddata(distances, qs, 
        #                                         distances_markers)
        perms_markers = scipy.interpolate.griddata(\
            distances, band_label_permutations, distances_markers, 
            method='nearest')
        
        # omegas_markers, _ = self.zerocalc.get_freqs_eigvecs_interpolated(\
        #     qs_markers, unit=unit, convention="c-type", include_nac=include_nac,
        #     clean_value=imaginary_cutoff)
        
        Ys = self.Y_interpolate(qs, include_nac=include_nac, freqs=omegas_calc, 
                                eigvecs=eigvecs, no_freqs=no_freqs,
                                imaginary_cutoff=imaginary_cutoff)
        Ys2 = np.real(Ys*Ys.conj())
        Ys2_sum = np.sum(Ys2, axis=1)
        omegas_markers = scipy.interpolate.griddata(distances, omegas, 
                                                    distances_markers)
        Ys2_markers = scipy.interpolate.griddata(distances, Ys2, 
                                                 distances_markers)
        Ys2_sum_markers = scipy.interpolate.griddata(distances, Ys2_sum, 
                                                     distances_markers)
        
        # Identify the degenerate phonon bands of the interpolated phonon bands
        degenerates = np.full((len(omegas_markers), number_of_bands), False)
        max_omega = np.max(omegas)
        degenerates[:,1:] = \
            np.array([(omegas_markers[:,i+1]-omegas_markers[:,i])/max_omega \
                      < degenerate_cutoff for i in range(number_of_bands-1)]).T
        
        # Sum over all degenerate branches in the second index of |Y|^2,
        # then put the result in the first degenerate branch
        Ys2_markers_degen = np.empty_like(Ys2_markers)
        Ys2_sum_markers_degen = np.empty_like(Ys2_sum_markers)
        for index, (Y2, Y2_sum) in enumerate(zip(Ys2_markers, Ys2_sum_markers)):
            degenerate = degenerates[index]
            degenerate_list = []
            temporary_list = []
            for index2 in range(number_of_bands-1, -1, -1):
                temporary_list.append(index2)
                if not degenerate[index2]:
                    degenerate_list.append(temporary_list)
                    temporary_list = []
            for indices in degenerate_list:
                Ys2_markers_degen[index, :, indices] = 0.
                Ys2_markers_degen[index, :, indices[0]] = np.sum(Y2[:, indices],
                                                                 axis=1)
                Ys2_sum_markers_degen[index, indices] = 0.
                Ys2_sum_markers_degen[index, indices[0]] = \
                    np.sum(Y2_sum[indices])

        # Normalize |Y|^2
        Y2_max_value = np.max(Ys2_markers_degen)
        Y2_sum_max_value = np.max(Ys2_sum_markers_degen)
        if Y2_norm_value is None:
            Y2_norm_value = Y2_max_value
        Ys2_norm = Ys2_markers_degen/Y2_norm_value
        if Y2_sum_norm_value is None:
            Y2_sum_norm_value = Y2_norm_value
        Ys2_sum_norm = Ys2_sum_markers_degen/Y2_sum_norm_value
        
        omegas_perm = np.array([omegas[i] @ band_label_permutations[i] 
                                for i in range(len(omegas))])
        omegas_markers_perm = np.array([omegas_markers[i] @ perms_markers[i] 
                                        for i in range(len(omegas_markers))])
        Ys2_perm = np.array([perms_markers[i].T @ Ys2_norm[i] @ perms_markers[i]
                             for i in range(len(Ys2_norm))])
        Ys2_sum_perm = np.array([Ys2_sum_norm[i] @ perms_markers[i] 
                                 for i in range(len(Ys2_sum_norm))])
        
        if subplots is None:
            fig_handles = []
            ax_handles = []
            for index in range(number_of_bands):
                fig, ax = plt.subplots(figsize=figsize)
                fig_handles.append(fig)
                ax_handles.append(ax)              
        else:
            fig, axs = plt.subplots(subplots[0], subplots[1], sharex=shareaxes,
                                    sharey=shareaxes, figsize=figsize)
            fig_handles = [fig]
            ax_handles = axs.flat
        plot_handles = []
        
        if plot_range is None:
            omega_min = np.min(omegas)
            omega_max = np.max(omegas)
            if omega_min < -self.zerocalc.convert_units(0.1, from_unit="THz", 
                                                        to_unit=unit):
                # Include the unstable phonon modes in the plot
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max)
            else:
                omega_min_scale, omega_max_scale = \
                    round_plot_range(omega_min, omega_max, clamp_min=0)
        else:
            omega_min_scale = plot_range[0]
            omega_max_scale = plot_range[1]
        
        for index, ax in enumerate(ax_handles):
            marker_sizes = marker_max_radius**2*Ys2_perm[:,index,:]
            
            plot_handles_this = \
                ax.plot(distances, np.delete(omegas_perm, index, axis=1), 
                        **plot_style, zorder=5)
            plot_handle_highlighted_line = \
                ax.plot(distances, omegas_perm[:,index], 
                        **plot_highlight_style, zorder=10)
            plot_handle_markers = \
                ax.scatter(distances_array, omegas_markers_perm, s=marker_sizes,
                           **marker_style, zorder=0)
            plot_handles_this.append(plot_handle_highlighted_line)
            plot_handles_this.append(plot_handle_markers)
            
            xaxis_ticks = np.append(distances[0::npoints], distances[-1])
            
            if set_title1:
                if no_freqs:
                    title1 = "$|\\tilde{Y}_{\\nu "+str(index)+\
                        ",z}(\\mathbf{q})|^2$"
                else:
                    title1 = "$|Y_{\\nu "+str(index)+",z}(\\mathbf{q})|^2$"
            ax.set_title(title1, size=text_sizes[2])
            ax.set_ylabel('Phonon frequencies (THz)', fontsize=text_sizes[1])
            ax.set_xticks(xaxis_ticks, xaxis_labels)
            ax.tick_params(axis='both', labelsize=text_sizes[0])
            ax.set_xlim(np.min(distances),np.max(distances))
            ax.set_ylim(omega_min_scale, omega_max_scale)
            
            # Plot major axis ticks
            for tick in xaxis_ticks:
                ax.axvline(x=tick, color='black', linewidth=0.5)
            
            # Plot unstable region in case of imaginary phonon frequencies
            if omega_min_scale < 0:
                ax.axhline(y=0.0, color='black', linewidth=1.0, 
                           linestyle="dashed")
                ax.add_patch(plt.Rectangle((distances[0], omega_min_scale), 
                                           distances[-1]-distances[0], 
                                           -omega_min_scale, fill=True, 
                                           color=(0.8, 0.8, 0.8), zorder=-10))
            
            plot_handles.append(plot_handles_this)
            if save_filename is not None and subplots is None:
                create_path(save_filename)
                fig_handles[index].tight_layout()
                fig_handles[index].savefig(save_filename+"_"+str(index)+".pdf")
        for fig in fig_handles:
            fig.tight_layout()
            fig.show()
        if save_filename is not None and subplots is not None:
            create_path(save_filename)
            fig_handles[0].savefig(save_filename+".pdf")
            
        fig, ax = plt.subplots()
        marker_sizes = marker_max_radius**2*Ys2_sum_perm
        plot_handles_this = \
            ax.plot(distances, omegas_perm, **plot_style, zorder=5)
        plot_handle_markers = \
            ax.scatter(distances_array, omegas_markers_perm, s=marker_sizes, 
                       **marker_style, zorder=0)
        plot_handles_this.append(plot_handle_markers)
        xaxis_ticks = np.append(distances[0::npoints], distances[-1])
        if set_title2:
            title2 = "$\\sum_{\\nu'} |Y_{\\nu \\nu',z}(\\mathbf{q})|^2$"
        ax.set_title(title2, size=text_sizes[2])
        ax.set_ylabel('Phonon frequencies (THz)', fontsize=text_sizes[1])
        ax.set_xticks(xaxis_ticks, xaxis_labels)
        ax.tick_params(axis='both', labelsize=text_sizes[0])
        ax.set_xlim(np.min(distances),np.max(distances))
        ax.set_ylim(omega_min_scale, omega_max_scale)
        # Plot major axis ticks
        for tick in xaxis_ticks:
            ax.axvline(x=tick, color='black', linewidth=0.5)
        # Plot unstable region in case of imaginary phonon frequencies
        if omega_min_scale < 0:
            ax.axhline(y=0.0, color='black', linewidth=1.0, linestyle="dashed")
            ax.add_patch(plt.Rectangle((distances[0], omega_min_scale), 
                                       distances[-1]-distances[0], 
                                       -omega_min_scale, fill=True, 
                                       color=(0.8, 0.8, 0.8), zorder=-10))
        fig.tight_layout()
        fig.show()
        if save_filename is not None:
            create_path(save_filename)
            fig.savefig(save_filename+"_summed.pdf")
        
        return Y2_max_value, Y2_sum_max_value, fig_handles, ax_handles, \
            plot_handles
    
class TomegaResults():
    """ Store and plot results of a Tomega calculation """

    def __init__(self, results_dict):
        """ TomegaResults(results_dict)

        Arguments
        ---------
        results_dict: dict or string
            If dict: Dictionary containing the results of a calculation
            If string: path to a .npz file generated
                by YCalculation.calculate_Tomega()
        """
        if isinstance(results_dict, str):
            self.results_dict = dict(np.load(results_dict))
        else:
            self.results_dict = results_dict
        self.q_mesh_size = int(self.results_dict["q_mesh_size"])
        self.unit = str(self.results_dict["unit"])
        self.include_nac = str(self.results_dict["include_nac"])
        self.sigma = float(self.results_dict["sigma"])
        self.moments = np.array(self.results_dict["moments"])
        self.moments_scaling_frequency = \
            float(self.results_dict["moments_scaling_frequency"])
        self.omega = np.array(self.results_dict["omega"])
        self.omega_max = float(self.results_dict["omega_max"])
        self.Tomega_LATO = np.array(self.results_dict["Tomega_LATO"])
        self.Tomega_LT = np.array(self.results_dict["Tomega_LT"])
        self.Tomega_AO = np.array(self.results_dict["Tomega_AO"])
        self.Tomega = np.array(self.results_dict["Tomega"])
        self.Tmoments_LATO = np.array(self.results_dict["Tmoments_LATO"])
        self.Tmoments_LT = np.array(self.results_dict["Tmoments_LT"])
        self.Tmoments_AO = np.array(self.results_dict["Tmoments_AO"])
        self.Tmoments = np.array(self.results_dict["Tmoments"])
        self.temperature = float(self.results_dict["temperature"])
    
    def to_dict(self):
        """ Return results data as a dict """
        return self.results_dict
    
    def save_npz(self, save_filename):
        """ Save all data in this object to a .npz file

        Arguments
        ---------
        save_filename: string
            Name of the file to save to
        """
        create_path(save_filename)
        np.savez(save_filename, q_mesh_size=self.q_mesh_size, unit=self.unit, 
                 include_nac=self.include_nac, sigma=self.sigma, 
                 moments=self.moments, 
                 moments_scaling_frequency=self.moments_scaling_frequency, 
                 omega=self.omega, omega_max=self.omega_max, 
                 Tomega_LATO=self.Tomega_LATO, Tomega_LT=self.Tomega_LT, 
                 Tomega_AO=self.Tomega_AO, Tomega=self.Tomega, 
                 Tmoments_LATO=self.Tmoments_LATO, Tmoments_LT=self.Tmoments_LT,
                 Tmoments_AO=self.Tmoments_AO, Tmoments=self.Tmoments,
                 temperature=self.temperature)
    
    def save_txt(self, save_filename, save_moments_filename=None,
                 resolution="LATO"):
        """ Save T(omega) and its moments to a .txt file

        This function saves the values of T(omega) and its moments
        to a text file. If a resolution is selected, also saves the
        separate longitudinal, acoustic, transverse, and/or optical
        contributions to the text file.

        Moments are made dimensionless with moments_scaling_frequency,
        which is also saved to the text file.

        Arguments
        ---------
        save_filename: string
            Name of the file to save T(omega) to
        save_moments_filename: string
            Name of the file to save the moments T_n to
            Default: None, moments are not saved to a file
        resolution: string
            Save the resolved contributions to T(omega) and its moments
            Choose from the following options:
            - "LATO": resolve longitudinal, acoustic, transverse, and
              optical
            - "LT": resolve longitudinal and transverse
            - "AO": resolve acoustic and optical
            - "none": only save the total contributions
            Default: "LATO"
        
        """
        match resolution:
            case "LATO":
                Tomega_resolutions = self.Tomega_LATO
                Tmoments_resolutions = self.Tmoments_LATO
                num_partials = len(Tomega_resolutions[0])
                labels = ["TA", "LA", "TO", "LO"]
            case "LT":
                Tomega_resolutions = self.Tomega_LT
                Tmoments_resolutions = self.Tmoments_LT
                num_partials = len(Tomega_resolutions[0])
                labels = ["T", "L"]
            case "AO":
                Tomega_resolutions = self.Tomega_AO
                Tmoments_resolutions = self.Tmoments_AO
                num_partials = len(Tomega_resolutions[0])
                labels = ["A", "O"]
            case "none":
                Tomega_resolutions = self.Tomega
                Tmoments_resolutions = self.Tmoments
                num_partials = 1
                labels = ["Tomega"]
            case _:
                warn_string = "resolution="+str(resolution)+\
                    "was not recognized."+\
                    "Expected 'LATO', 'LT', 'AO', or 'none'."+\
                    "A file with LATO resolution was saved."
                warnings.warn(warn_string)
                Tomega_resolutions = self.Tomega_LATO
                Tmoments_resolutions = self.Tmoments_LATO
                labels = ["TA", "LA", "TO", "LO"]
        if resolution=='none':
            full_data_array = \
                np.transpose(np.array([self.omega,self.Tomega]))
            header = "  Frequency ("+self.unit+")"+" "*(9-len(self.unit))+\
                     "    T(omega), total      "
            moments_data_array = \
                np.transpose(np.array([self.moments,self.Tmoments]))
            header_moments = "Moments scaling frequency: "+\
                str(self.moments_scaling_frequency)+" "+self.unit+"\n"\
                "       Moment n                T_n, total       "
        else:
            num_contributions = int(num_partials*(num_partials+1)/2)
            T_contributions = np.zeros((num_contributions, len(self.omega)))
            T_moments_contributions = np.zeros((num_contributions, 
                                                len(self.moments)))
            contribution_labels = ["total"]
            count = 0
            for index1 in range(0, num_partials):
                T_contributions[count] = Tomega_resolutions[:, index1, index1]
                T_moments_contributions[count] = \
                    Tmoments_resolutions[:, index1, index1]
                contribution_labels.append(labels[index1]+"-"+labels[index1])
                count += 1
                for index2 in range(index1+1, num_partials):
                    T_contributions[count] = \
                        Tomega_resolutions[:, index1, index2]+\
                        Tomega_resolutions[:, index2, index1]
                    T_moments_contributions[count] = \
                        Tmoments_resolutions[:, index1, index2] \
                        + Tmoments_resolutions[:, index2, index1]
                    contribution_labels.append(labels[index1]+"-"+\
                                            labels[index2])
                    count += 1
            full_data_array = \
                np.transpose(np.append(np.array([self.omega,self.Tomega]), 
                                    T_contributions, axis=0))
            header = "  Frequency ("+self.unit+")"+" "*(9-len(self.unit))
            for name in contribution_labels:
                header += "    T(omega), "+name+" "*(11-len(name))
            moments_data_array = \
                np.transpose(np.append(np.array([self.moments,self.Tmoments]), 
                                    T_moments_contributions, axis=0))
            header_moments = "Moments scaling frequency: "+\
                str(self.moments_scaling_frequency)+" "+self.unit+"\n"\
                "       Moment n         "
            for name in contribution_labels:
                header_moments += "       T_n, "+name+" "*(13-len(name))
        create_path(save_filename)
        np.savetxt(save_filename+".txt", full_data_array,
                header=header)
        if save_moments_filename is not None:
            create_path(save_moments_filename)
            np.savetxt(save_moments_filename+".txt", moments_data_array,
                    header=header_moments)

    def plot(self, resolution="LATO", save_filename=None, inset_moment=-0.5, 
             inset_bounding_box=[0.79,0.15,0.1,0.8], text_sizes=(13, 15, 16),
             title=None, hatchplot_style=None): 
        """ Plot T(omega) and its LATO components

        Arguments
        ---------
        resolution: str
            Choose which components of T(omega) should be resolved
            on the figure
            Choose from the following options:
            - "LATO": resolve longitudinal, acoustic, transverse, and
              optical
            - "LT": resolve longitudinal and transverse
            - "AO": resolve acoustic and optical
            Default: "LATO"
        save_filename: str
            Save all figures with the given common filename, 
            in .pdf format
            The filename should not include the extension ".pdf"
            Default: None, figure is not saved to a file
        inset_moment: real
            Moment used to calculate data for the inset figure
            Note: moments must contain the value of inset_moment in
            order to plot the inset figure, otherwise a warning is 
            raised and a legend is plotted instead 
            Default: -0.5
        inset_bounding_box: List of 4 ints, or None
            Bounding box of the inset figure. If equal to None, plot
            a legend instead instead of an inset figure.
            Default: [0.79, 0.15, 0.1, 0.8]
        text_sizes: tuple of 3 ints
            Font sizes for small, medium and large text in the figure
                -Small text: axis ticks
                -Medium text: axis labels
                -Large text: title
            Default: (13, 15, 16)
        title: str
            Title for the figure
            Default: "1-electron-2-phonon spectral function"
        hatchplot_styles: List of 10 dicts
            Hatchings and colors to be used for the stackplot of LATO
            contributions. Each dict has 4 keys:
                - hatch: hatching style from Matplotlib
                - color1: background color for the hatching
                - color2: color of the hatching pattern
            
            Default:
            - For resolution="LATO":
                hatchplot_styles = [
                    dict(color1=colorTA, color2=colorTA, hatch='//'),
                    dict(color1=colorTA, color2=colorLA, hatch='//'),
                    dict(color1=colorTA, color2=colorTO, hatch='\\\\'),
                    dict(color1=colorTA, color2=colorLO, hatch='//'),
                    dict(color1=colorLA, color2=colorLA, hatch='//'),
                    dict(color1=colorLA, color2=colorTO, hatch='//'),
                    dict(color1=colorLA, color2=colorLO, hatch='\\\\'),
                    dict(color1=colorTO, color2=colorTO, hatch='//'),
                    dict(color1=colorTO, color2=colorLO, hatch='//'),
                    dict(color1=colorLO, color2=colorLO, hatch='//'),
                ]
            where colorTA = (0.8, 0.5, 0.5), colorLA = (0.4, 0.7, 0.4),
            colorTO = (0.4, 0.4, 0.8), and colorLO = (0.3, 0.3, 0.3)
            - For resolution="LT":
                hatchplot_style = [
                    dict(color1=colorT, color2=colorT, hatch='//'),
                    dict(color1=colorT, color2=colorL, hatch='//'),
                    dict(color1=colorL, color2=colorL, hatch='//'),
                ]
            where colorT = (0.8, 0.5, 0.5) and colorL = (0.4, 0.7, 0.4)
            - For resolution="AO":
                hatchplot_style = [
                    dict(color1=colorA, color2=colorA, hatch='//'),
                    dict(color1=colorA, color2=colorO, hatch='//'),
                    dict(color1=colorO, color2=colorO, hatch='//'),
                ]
            where colorA = (0.8, 0.5, 0.5) and colorO = (0.4, 0.4, 0.8)

        Returns
        -------
        fig: figure handle
        ax: axis handle
        plot_handle: handle to the plot data

        """
        
        match resolution:
            case "LATO":
                Tomega_resolutions = self.Tomega_LATO
                Tmoments_resolutions = self.Tmoments_LATO
                labels = ["TA", "LA", "TO", "LO"]
                if hatchplot_style is None:
                    colorTA = (0.8, 0.5, 0.5)
                    colorLA = (0.4, 0.7, 0.4)
                    colorTO = (0.4, 0.4, 0.8)
                    colorLO = (0.3, 0.3, 0.3)
                    hatchplot_style = [
                        dict(color1=colorTA, color2=colorTA, hatch='//'),
                        dict(color1=colorTA, color2=colorLA, hatch='//'),
                        dict(color1=colorTA, color2=colorTO, hatch='\\\\'),
                        dict(color1=colorTA, color2=colorLO, hatch='//'),
                        dict(color1=colorLA, color2=colorLA, hatch='//'),
                        dict(color1=colorLA, color2=colorTO, hatch='//'),
                        dict(color1=colorLA, color2=colorLO, hatch='\\\\'),
                        dict(color1=colorTO, color2=colorTO, hatch='//'),
                        dict(color1=colorTO, color2=colorLO, hatch='//'),
                        dict(color1=colorLO, color2=colorLO, hatch='//'),
                    ]
            case "LT":
                Tomega_resolutions = self.Tomega_LT
                Tmoments_resolutions = self.Tmoments_LT
                labels = ["T", "L"]
                if hatchplot_style is None:
                    colorT = (0.8, 0.5, 0.5)
                    colorL = (0.4, 0.7, 0.4)
                    hatchplot_style = [
                        dict(color1=colorT, color2=colorT, hatch='//'),
                        dict(color1=colorT, color2=colorL, hatch='//'),
                        dict(color1=colorL, color2=colorL, hatch='//'),
                    ]
            case "AO":
                Tomega_resolutions = self.Tomega_AO
                Tmoments_resolutions = self.Tmoments_AO
                labels = ["A", "O"]
                if hatchplot_style is None:
                    colorA = (0.8, 0.5, 0.5)
                    colorO = (0.4, 0.4, 0.8)
                    hatchplot_style = [
                        dict(color1=colorA, color2=colorA, hatch='//'),
                        dict(color1=colorA, color2=colorO, hatch='//'),
                        dict(color1=colorO, color2=colorO, hatch='//'),
                    ]
            case _:
                warn_string = "resolution="+str(resolution)+\
                    "was not recognized. Expected 'LATO', 'LT', or 'AO'."+\
                    "A figure with LATO resolution was made."
                warnings.warn(warn_string)
                Tomega_resolutions = self.Tomega_LATO
                Tmoments_resolutions = self.Tmoments_LATO
                labels = ["TA", "LA", "TO", "LO"]
                if hatchplot_style is None:
                    colorTA = (0.8, 0.5, 0.5)
                    colorLA = (0.4, 0.7, 0.4)
                    colorTO = (0.4, 0.4, 0.8)
                    colorLO = (0.3, 0.3, 0.3)
                    hatchplot_style = [
                        dict(color1=colorTA, color2=colorTA, hatch='//'),
                        dict(color1=colorTA, color2=colorLA, hatch='//'),
                        dict(color1=colorTA, color2=colorTO, hatch='\\\\'),
                        dict(color1=colorTA, color2=colorLO, hatch='//'),
                        dict(color1=colorLA, color2=colorLA, hatch='//'),
                        dict(color1=colorLA, color2=colorTO, hatch='//'),
                        dict(color1=colorLA, color2=colorLO, hatch='\\\\'),
                        dict(color1=colorTO, color2=colorTO, hatch='//'),
                        dict(color1=colorTO, color2=colorLO, hatch='//'),
                        dict(color1=colorLO, color2=colorLO, hatch='//'),
                    ]

        # Find the first index where moments = inset_moment
        moments_inset_indices = \
            np.where(np.abs(self.moments-inset_moment)<=1e-10)[0]
        if len(moments_inset_indices) == 0:
            warn_string = "inset_moment was not found in moments."+\
             "A legend was plotted instead of the inset figure."
            warnings.warn(warn_string)
            inset_bounding_box = None
        else:
            moments_inset_index = moments_inset_indices[0]
        
        num_partials = len(Tomega_resolutions[0])
        num_contributions = int(num_partials*(num_partials+1)/2)
        T_contributions = np.zeros((num_contributions, len(self.omega)))
        T_moments_contributions = np.zeros((num_contributions, 
                                            len(self.moments)))
        contribution_labels = []
        count = 0
        for index1 in range(0, num_partials):
            T_contributions[count] = Tomega_resolutions[:, index1, index1]
            T_moments_contributions[count] = \
                Tmoments_resolutions[:, index1, index1]
            contribution_labels.append(labels[index1]+"-"+labels[index1])
            count += 1
            for index2 in range(index1+1, num_partials):
                T_contributions[count] = Tomega_resolutions[:, index1, index2]+\
                    Tomega_resolutions[:, index2, index1]
                T_moments_contributions[count] = \
                    Tmoments_resolutions[:, index1, index2] \
                    + Tmoments_resolutions[:, index2, index1]
                contribution_labels.append(labels[index1]+"-"+\
                                           labels[index2])
                count += 1
        if inset_bounding_box is not None:
            T_mhalf_relative = \
                T_moments_contributions[:,moments_inset_index] \
                / np.sum(T_moments_contributions[:,moments_inset_index])
        else:
            # We don't need T_mhalf_relative
            T_mhalf_relative = np.empty((num_contributions,))
        
        fig, ax = plt.subplots()
        total_stack = np.zeros_like(self.omega)
        plot_handles = []
        text_handles = []
        if inset_bounding_box is not None:
            ax2 = ax.inset_axes(inset_bounding_box)
            total_stack2 = 0
            plot_handles2 = []
        for index, (data, T_mhalf, style, label) \
            in enumerate(zip(T_contributions, T_mhalf_relative,
                            hatchplot_style, contribution_labels)):
            plot_handle = \
                ax.fill_between(self.omega, total_stack + data, y2=total_stack, 
                                zorder=-index, linewidth=0, label=label,
                                hatch=style['hatch'], fc=style['color1'], 
                                edgecolor=style['color2'])
            ax.plot(self.omega, total_stack + data, zorder=0, color='k', 
                    linewidth=0.5)
            total_stack += data
            plot_handles.append(plot_handle)
            if inset_bounding_box is not None:
                plot_handle2 = ax2.bar(0.0, T_mhalf, 1.0, label=label, 
                                    zorder=-index, bottom=total_stack2, 
                                    linewidth=0, hatch=style['hatch'], 
                                    fc=style['color1'], 
                                    edgecolor=style['color2'])
                ax2.plot(np.array([-0.5,0.5]), 
                            np.tile(total_stack2+T_mhalf, 2), 
                            zorder=0, color='k', linewidth=0.5)
                t_han = ax2.text(0.51, total_stack2 + 0.5*T_mhalf, 
                            label, fontsize=12, horizontalalignment="left", 
                            verticalalignment="center_baseline")
                total_stack2 += T_mhalf
                plot_handles2.append(plot_handle2)
                text_handles.append(t_han)
        plot_handle_total, = ax.plot(self.omega, self.Tomega, 
                                        color="black", label="Total")
        plot_handles.append(plot_handle_total)
        
        if title is None:
            title = "1-electron-2-phonon spectral function"
        ax.set_title(title, fontsize=text_sizes[2])
        ax.set_xlabel("Frequency ("+self.unit+")", fontsize=text_sizes[1])
        ax.set_ylabel("$\\mathcal{T}(\\omega)$", fontsize=text_sizes[1])
        xmin, xmax = round_plot_range(0, self.omega_max, clamp_min=0)
        ymin, ymax = round_plot_range(0, np.max(self.Tomega), clamp_min=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', labelsize=text_sizes[0])
        if inset_bounding_box is None:
            ax.legend(handles=plot_handles[::-1], fontsize=text_sizes[0])
        else:
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_ylim( 0.0, 1)
            ax2.set_ylabel("$\\mathcal{T}_{"+str(inset_moment)+\
                            "}$ contribution", fontsize=text_sizes[0])
            ax2.tick_params(axis='both', labelsize=text_sizes[0],
                            left=False, labelleft=False, bottom=False, 
                            labelbottom=False)
            bboxes = [txt_han.get_window_extent() \
                      .transformed(ax2.transData.inverted()) 
                      for txt_han in text_handles]
            
            # Calculate new label positions so that the labels don't overlap
            text_heights = [bbox.y1-bbox.y0 for bbox in bboxes]
            text_middles = np.array([0.5*(bbox.y0+bbox.y1) for bbox in bboxes])
            for _ in range(50): # perform at most 50 steps
                collision_detected = False
                for index in range(len(text_middles)-1):
                    diff = text_middles[index+1]-text_middles[index] \
                        - 0.5*(text_heights[index+1] + text_heights[index])
                    if diff < 0:
                        # If two labels overlap, push them both equally far away
                        text_middles[index] += 0.6*diff
                        text_middles[index+1] -= 0.6*diff
                        collision_detected = True
                if not collision_detected:
                    break
            for txt_han, y_pos in zip(text_handles, text_middles):
                txt_han.set_y(y_pos)

        fig.tight_layout()
        fig.show()
        if save_filename is not None:
            create_path(save_filename)
            fig.savefig(save_filename+".pdf")
        return fig, [ax, ax2], plot_handles
    
