"""
High-symmetry paths and labels of special points for different types
of Brillouin lattices, including an option to break z-symmetry.
Data for high-symmetry paths obtained from:
W. Setyawan and S. Curtarolo, Comp. Mat. Sci. 49, 2, 299-312 (2010)
doi.org/10.1016/j.commatsci.2010.05.010

Exported functions
------------------
get_path_and_labels: data for high-symmetry paths in the Brillouin zone

Written by Matthew Houtput (matthew.houtput@uantwerpen.be)
"""

def get_path_and_labels(lattice_type, break_z=False):
    """ Path and labels of high-symmetry paths in the Brillouin zone
    
    Arguments
    ---------
    lattice_type: str
        Three-letter code for the type of Brillouin lattice
        Currently implemented:
            - "CUB", simple cubic
            - "FCC", face-centered cubic
            - "BCC", body-centered cubic
    break_z: bool
        - True: return high-symmetry path for material with broken
                z-symmetry (e.g. by an electric field)
        - False: return the usual high-symmetry path
    
    Returns
    -------
    path: list of list of list of real
        High-symmetry path, written in direct coordinates in the same
        conventions as PhonoPy
            - First level: list of connected path segments
            - Second level: list of points that mark path segments
            - Third level: direct coordinates of points
    path_labels: list of str
        List of names of the edge points of the path, in order,
        written in LaTeX markup.
        If break_z == True, labels of the high-symmetry structure
        are used, adding subscripts _1, _2, ... if necessary
    
    """

    match lattice_type:
        case "CUB":
            # Simple cubic lattice
            if break_z:                           
                path = [                                # CUB label | TET label                       
                            [                           # ---------------------
                                [0.0, 0.0, 0.0],        # Gamma     | Gamma
                                [0.0, 0.5, 0.0],        # X         | X
                                [0.5, 0.5, 0.0],        # M         | M
                                [0.0, 0.0, 0.0],        # Gamma     | Gamma
                                [0.0, 0.0, 0.5],        # X1        | Z
                                [0.0, 0.5, 0.5],        # M1        | R
                                [0.5, 0.5, 0.5],        # R         | A
                                [0.0, 0.0, 0.5],        # X1        | Z
                            ],
                            [
                                [0.0, 0.5, 0.0],        # X         | X
                                [0.0, 0.5, 0.5],        # M1        | R
                            ],
                            [
                                [0.5, 0.5, 0.0],        # M         | M
                                [0.5, 0.5, 0.5]         # R         | A
                            ]
                        ]
                path_labels = ["$\\Gamma$", "X", "M", "$\\Gamma$", "X$_1$", 
                               "M$_1$", "R", "X$_1$", "X", "M$_1$", "M", "R"]
            else:
                path = [
                            [
                                [0.0, 0.0, 0.0],        # Gamma
                                [0.0, 0.5, 0.0],        # X
                                [0.5, 0.5, 0.0],        # M
                                [0.0, 0.0, 0.0],        # Gamma
                                [0.5, 0.5, 0.5],        # R
                                [0.0, 0.5, 0.0],        # X
                            ],
                            [
                                [0.5, 0.5, 0.0],        # M
                                [0.5, 0.5, 0.5]         # R
                            ]
                        ]
                path_labels = ["$\\Gamma$", "X", "M", "$\\Gamma$", "R", "X",
                               "M", "R"]
        case "FCC":
            # Face-centered cubic lattice
            if break_z:                           
                path = [                                # FCC label | BCT label                       
                            [                           # ---------------------
                                [0.0  , 0.0  , 0.0  ],  # Gamma     | Gamma
                                [0.5  , 0.0  , 0.5  ],  # X         | X
                                [0.5  , 0.25 , 0.75 ],  # W         | Y
                                [0.375, 0.375, 0.75 ],  # K         | Sigma
                                [0.0  , 0.0  , 0.0  ],  # Gamma     | Gamma
                                [0.5  , 0.5  , 0.0  ],  # X1        | Z
                                [0.625, 0.625, 0.25 ],  # U1        | Sigma1
                                [0.5  , 0.5  , 0.5  ],  # L         | N
                                [0.75 , 0.25 , 0.5  ],  # W1        | P
                                [0.75 , 0.375, 0.375],  # K1        | -
                                [0.75 , 0.5  , 0.25 ],  # W2        | Y1
                                [0.5  , 0.5  , 0.0  ],  # X1        | Z
                            ],
                            [
                                [0.5  , 0.0  , 0.5  ],  # X         | X
                                [0.75 , 0.25 , 0.5  ]   # W1        | P
                            ]
                        ]
                path_labels = ["$\\Gamma$", "X", "W", "K", "$\\Gamma$",
                               "X$_1$", "U$_1$", "L", "W$_1$", "K$_1$", 
                               "W$_2$", "X$_1$", "X", "W$_1$"]
            else:
                path = [
                            [
                                [0.0  , 0.0  , 0.0  ],  # Gamma
                                [0.5  , 0.0  , 0.5  ],  # X
                                [0.5  , 0.25 , 0.75 ],  # W
                                [0.375, 0.375, 0.75 ],  # K
                                [0.0  , 0.0  , 0.0  ],  # Gamma
                                [0.5  , 0.5  , 0.5  ],  # L
                                [0.625, 0.25 , 0.625],  # U
                                [0.5  , 0.25 , 0.75 ],  # W
                                [0.5  , 0.5  , 0.5  ],  # L
                                [0.375, 0.375, 0.75 ]   # K
                            ],
                            [
                                [0.625, 0.25 , 0.625],  # U
                                [0.5  , 0.0  , 0.5  ]   # X
                            ]
                        ]
                path_labels = ["$\\Gamma$", "X", "W", "K", "$\\Gamma$", "L",
                               "U", "W", "L", "K", "U", "X"]
        case "BCC":
            # Body-centered cubic lattice
            if break_z:                           
                path = [                                # BCC label | BCT label                       
                            [                           # ---------------------
                                [ 0.0 , 0.0 , 0.0 ],    # Gamma     | Gamma
                                [ 0.0 , 0.0 , 0.5 ],    # N         | X
                                [-0.5 , 0.5 , 0.5 ],    # H1        | M
                                [ 0.0 , 0.0 , 0.0 ],    # Gamma     | Gamma
                                [ 0.5 , 0.5 ,-0.5 ],    # H2        | Z
                                [ 0.25, 0.25, 0.25],    # P         | P
                                [ 0.0 , 0.5 , 0.0 ],    # N1        | N
                                [-0.5 , 0.5 , 0.5 ],    # H1        | Z1
                            ],
                            [
                                [ 0.0 , 0.0 , 0.5 ],    # N         | X
                                [ 0.25, 0.25, 0.25]     # P         | P
                            ]
                        ]
                path_labels = ["$\\Gamma$", "N", "H$_1$", "$\\Gamma$", "H$_2$",
                               "P", "N$_1$", "H$_1$", "N", "P"]
            else:
                path = [
                            [
                                [ 0.0 , 0.0 , 0.0 ],    # Gamma
                                [ 0.5 ,-0.5 , 0.5 ],    # H
                                [ 0.0 , 0.0 , 0.5 ],    # N
                                [ 0.0 , 0.0 , 0.0 ],    # Gamma
                                [ 0.25, 0.25, 0.25],    # P
                                [ 0.0 , 0.0 , 0.0 ],    # H
                            ],
                            [
                                [ 0.25, 0.25, 0.25],    # P
                                [ 0.0 , 0.0 , 0.5 ]     # N
                            ]
                        ]
                path_labels = ["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H",
                               "P", "N"]
        case _:
            raise NameError(("Unknown lattice_type '"+str(lattice_type)+"': \n"
                             "please choose from the implemented list \n" 
                             "{'CUB', 'FCC', 'BCC'}"))
    return path, path_labels
            