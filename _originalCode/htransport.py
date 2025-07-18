"""
High Temperature Air Transport Properties
 - Calculates transport properties of based on given cantera 
   gas object, gas temperature and pressure.
   
 - Dissocation effects on specific heat and thermal conduct-
   ivity is included.
   
Assumptions:
 - Chemical and thermodynamic equilibrium
 
 - Ideal gas

Author:     Dogan Akcakaya
Date:       22.08.2023
"""

import cantera as ct
import numpy as np
import os
from numpy import *

# defaul initilization
script_dir = os.path.dirname(os.path.realpath(__file__))
yaml_file_path = os.path.join(script_dir, "airNASA9-transport.yaml")
htair = ct.Solution(yaml_file_path)

def initialize():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_file_path = os.path.join(script_dir, "airNASA9-transport.yaml")
    htair = ct.Solution(yaml_file_path)

def get_transport(T, p, gas=htair):
    initialize()
    # set state
    gas.TP = T, p
    gas.equilibrate('TP')
    
    p0 = gas.P          # pressure
    h0 = gas.h          # enthalpy
    T0 = gas.T          # temperature
    X0 = gas.X          # molar concentration
    Y0 = gas.Y          # mass concentration
    d0 = gas.density    # density
    v0 = gas.viscosity  # dynamic viscosity
    k0 = gas.thermal_conductivity

    binary_coeffs = gas.mix_diff_coeffs # mixture binary coeffs
    
    diff = np.zeros(shape=(8,))
    for i in range (0,8):
        diff[i] = (1-X0[i]) / np.sum(X0 / binary_coeffs[i])
        
    hpartial = gas.partial_molar_enthalpies / gas.molecular_weights   # partial molar enthalpies

    T1 = T0*1.000001    # perturbate temperature
    
    # set perturbation state
    gas.TP = T1, p0
    gas.equilibrate('TP')

    h1 = gas.h              # total perturbated enthalpy of mixture
    X1 = gas.X              # perturbated molar concentration
    Y1 = gas.Y              # perturbated mass concentration
    
    # this one is unnecessary??
    cp = (h1 - h0)/(T1-T0)  # calculation of specific heat via numerical derivative
    
    # reactive component of thermal conductivity due to diffusion
    kr = d0 * np.sum(diff * hpartial * (Y1 - Y0) / (T1-T0))
    k = k0 + kr   # total conductivity
    
    # viscosity
    v = v0
    
    return(cp, k, v)