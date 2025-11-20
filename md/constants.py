import numpy as np


def get_lattice_constant(id):
    if id == "Ag":
        a = 4.09
    elif id == "Al":
        a = 4.05
    elif id == "Au":
        a = 4.08
    elif id == "Cu":
        a = 3.615
    elif id == "Ni":
        a = 3.52
    elif id == "Pb":
        a = 4.95
    elif id == "Pd":
        a = 3.89
    elif id == "Pt":
        a = 3.92
    else: 
        print("ERROR: Invalid material entered.")
        return -1
    return a

def get_amu(id):
    if id == "Ag":
        amu = 107.868
    elif id == "Al":
        amu = 26.981
    elif id == "Au":
        amu = 196.966
    elif id == "Cu":
        amu = 63.546
    elif id == "Ni":
        amu = 58.693
    elif id == "Pb":
        amu = 207.2
    elif id == "Pd":
        amu = 106.42
    elif id == "Pt":
        amu = 195.08
    else: 
        print("ERROR: Invalid material entered.")
        return -1
    return amu

# Returns sigma for LJ potential
# Needed rescaling
def get_sigma(id):
    if id == "Ag":
        sigma = 2.955 / (2**(1/6))
    elif id == "Al":
        sigma = 2.925 / (2**(1/6))
    elif id == "Au":
        sigma = 2.951 / (2**(1/6))
    elif id == "Cu":
        sigma = 2.616 / (2**(1/6))
    elif id == "Ni":
        sigma = 2.552 / (2**(1/6))
    elif id == "Pb":
        sigma = 3.565 / (2**(1/6))
    elif id == "Pd":
        sigma = 2.819 / (2**(1/6))
    elif id == "Pt":
        sigma = 2.845 / (2**(1/6))
    else: 
        print("ERROR: Invalid material entered.")
        return -1
    return sigma
    
# Returns eps for LJ potential
# Units came in kcal/mol so need to convert to eV
def get_eps(id):
    if id == "Ag":
        eps = 4.56 * 0.043364115
    elif id == "Al":
        eps = 4.02 * 0.043364115
    elif id == "Au":
        eps = 5.29 * 0.043364115
    elif id == "Cu":
        eps = 4.72 * 0.043364115
    elif id == "Ni":
        eps = 5.65 * 0.043364115
    elif id == "Pb":
        eps = 2.93 * 0.043364115
    elif id == "Pd":
        eps = 6.15 * 0.043364115
    elif id == "Pt":
        eps = 7.80 * 0.043364115
    else: 
        print("ERROR: Invalid material entered.")
        return -1
    return eps