"""
 Description:
 - Heat transfer correlations for high velocity external flows
 - Gases with variables properties
 - Corrected using property ratio
 - Reference: Kays and Crawford (1993)
    
 Assumptions:
 - Temperature formulation: cp is constant (%10 error expected up to 800-900K)
 - No pressure gradient across the boundary layer of in flow direction
    - Pressure across boundary layer is zero for nonreacting flows. For hypersonic,
      pressure graident may not be zero but still expected to be very small for
      low level reacting flows. (Valid for body heating)
 - Laminar
 
 Author:    Dogan AKCAKAYA
 Date:      24.08.2023
"""

# dependicies
import os
import numpy as np
import cantera as ct
import htshock2 as ht
import htransport as htp
import math

# defaul initilization
script_dir = os.path.dirname(os.path.realpath(__file__))
yaml_file_path = os.path.join(script_dir, "airNASA9-transport.yaml")
htair = ct.Solution(yaml_file_path)

def initialize():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_file_path = os.path.join(script_dir, "airNASA9-transport.yaml")
    htair = ct.Solution(yaml_file_path)

# Property Ratio Method with variable properties
# gas: cantera gas object
# Mf, Tf, pf: local free stream mach, static temperature, pressure
# x: local location from leading edge for local reynolds
def propertyRatioHT(Mf, Tf, pf, x, wedge='no', msg='no'):
    initialize()
    x = np.atleast_1d(x)
    
    # set the state in case it isn't
    gas = ct.Solution('airNASA9-transport.yaml')
    gas.TP = Tf, pf
    gas.equilibrate('TP')

    # Define a wall temperature
    # Dependence on wall temperature is weak and lower temperature results with higher coefficients
    # therefore a low and constant wall temperature expected to yield insignificant errors
    Tw = 273.15 + 100
    
    # equilibrium speed of sound
    af = ht.eqsound(gas)
    
    # free stream velocity
    uf = Mf * af
    
    # free stream enthalpy
    hf = gas.h
    
    # Get thermo properties
    rho = gas.density
    cp = gas.cp
    
    # Get transport properties
    cp, k, mu = htp.get_transport(Tf, pf) # mu is dynamic viscosity
    
    # length array
    ndatax = x.shape[0]
    output = np.zeros(shape=(ndatax*4+1, ))
    
    for i in range(0, ndatax):
        ## Constant property solution
        # Dimensionless groups based on free stream temperature
        Rex_CP = rho * uf * x[i] / mu                 # local reynolds number based on free stream
        Pr_CP = mu * cp / k                           # prandtl number based on free stream
        
        
        if Rex_CP <= 1.0E6: # laminar
            if msg == 'yes':
                print('laminar')
            Nux_CP = math.sqrt(3) * 0.332 * Pr_CP**(1/3) * Rex_CP**(1/2) # local nusselt number based on free stream
            if wedge == 'yes': 
                Nux_CP = Nux_CP / math.sqrt(3)
            # Stx_CP = Nux_CP / Pr_CP / Rex_CP
            htc_CP = Nux_CP * k / x[i]                           # local heat transfer coefficient
            
            # adiabatic wall temperature
            rc = Pr_CP**(1/2)                                    # recovery factor
            haw = hf + rc * uf**2 / 2                            # adibatic wall enthalpy
            gas.HP = haw, pf
            gas.equilibrate('HP')
            Taw = gas.T                                          # adiabatic wall temperature
            
            # corrected
            htc = htc_CP * (Tw/Tf)**(-0.08) * (Taw/Tf)**(-0.04)
            
        else:   # turbulent
            if msg == 'yes':
                print('turbulent')
            Nux_CP = 1.1 * 0.0287 * Pr_CP**(0.6) * Rex_CP**(0.8)
            if wedge == 'yes':
                Nux_CP = Nux_CP / 1.1
            htc_CP = Nux_CP * k / x[i] 
            
            rc = Pr_CP**(1/3)                                    # recovery factor
            haw = hf + rc * uf**2 / 2                            # adibatic wall enthalpy
            gas.HP = haw, pf
            gas.equilibrate('HP')
            Taw = gas.T                                          # adiabatic wall temperature
            
            # corrected
            htc = htc_CP * (Taw/Tf)**(-0.6) * (Tw/Taw)**(-0.4)

        # rc = Pr_CP**(1/3)                             # recovery factor
        # Taw = Tf + rc * uf**2 / 2 / cp                # adibatic wall temperature
        # form output
        output[4*i], output[4*i+1], output[4*i+2], output[4*i+3] = x[i], Rex_CP, Taw, htc
    
    output[-1] = Taw
    return(output)   
   
# Pass parameters after normal shock
def stagnationHT(Mf, Tf, pf, pfree, Rnose):
    initialize()
    # pfree free stream temperature
    
    # set the state in case it isn't
    gas = ct.Solution('airNASA9-transport.yaml')
    gas.TP = Tf, pf
    gas.equilibrate('TP')
    
    # Define a wall temperature
    # Dependence on wall temperature is weak and lower temperature results with higher coefficients
    # therefore a low and constant wall temperature expected to yield insignificant errors
    Tw = 273.15 + 100
    
    # equilibrium speed of sound
    af = ht.eqsound(gas)
    
    # free stream velocity
    uf = Mf * af
    
    # Get thermo properties
    rhof = gas.density
    
    # isentropic stagnation state
    rho_stag, p_stag, T_stag = ht.stagnation(Mf, uf, rhof, pf, Tf)
    
    gas.TP = T_stag, p_stag
    gas.equilibrate('TP')
    
    h_stag = gas.h
    
    # transport properties @stagnation
    cp_stag, k_stag, mu_stag = htp.get_transport(T_stag, p_stag) # mu is dynamic viscosity
    
    # properties at @wall
    gas.TP = Tw, p_stag
    gas.equilibrate('TP')
    
    h_w = gas.h
    rho_w = gas.density
    
    cp_w, k_w, mu_w = htp.get_transport(Tw, p_stag) # mu is dynamic viscosity
    Pr_w = mu_w * cp_w / k_w

    # velocity gradient
    u_grad = 1.0 / Rnose * math.sqrt(2.0*(p_stag-pfree)/rho_stag)
   
    # heat transfer coefficient
    htc_stag = 0.7 *(rho_stag*mu_stag)**0.44 * (rho_w*mu_w)**0.06 * math.sqrt(u_grad) * (h_stag - h_w)/ (T_stag - Tw)
    htc_stag2 = 0.94 *(rho_stag*mu_stag)**0.4 * (rho_w*mu_w)**0.1 * math.sqrt(u_grad) * (h_stag - h_w)/ (T_stag - Tw)
    # htc_stag = 5.1564 * 10**(-5) / math.sqrt(Rnose) * (0.04028530484047352)**0.5 * 1375**(3.15) / (T_stag - Tw)
    output = np.zeros(shape=(3, )) 
    output[0], output[1], output[2] = T_stag, htc_stag, htc_stag2
    return(output)


"""
M2, P2, T2, rho2, u2 = M_inf.copy(), P_inf.copy(), T_inf.copy(), rho_inf.copy(), u_inf.copy()
mu2, cond2, cp2, a2, h2 = zeros(sizeveri), zeros(sizeveri), zeros(sizeveri), zeros(sizeveri), zeros(sizeveri)

for i in r_[0:sizeveri]:
    if M_inf[i] > 1.1:
        M2[i],T2[i],P2[i],rho2[i],u2[i] = discontinuity.nshock(M_inf[i],T_inf[i],P_inf[i],rho_inf[i],u_inf[i])

a2 = sqrt(401.856 * T2)

for i in r_[0:sizeveri]:
    mu2[i], cond2[i], cp2[i], h2[i] = discontinuity.prop(T2[i])

H_r, T_r = zeros(sizeveri), zeros(sizeveri)
H_r =  h2 + 0.5*u2**2

for i in r_[0:sizeveri]:
    T_r[i] = discontinuity.h_to_T(H_r[i])

P_stag, rho_stag, T_stag, mu_stag = zeros(sizeveri), zeros(sizeveri), zeros(sizeveri), zeros(sizeveri)
P_stag = P2*(1. + 0.2*M2**2)**3.5
rho_stag = rho2*(1.+ 0.2*M2**2)**2.5
T_stag = T2 * (1.+ 0.2*M2**2)
mu_stag = 3.7978318E-7 * T_stag**0.68293666

rho_w = P_stag / (287.04 * T_w)
mu_w = 3.7978318E-7 * T_w**0.68293666
cp_w = 1030.39 + T_w*(-2.6338079E-1+T_w*(7.5238345E-4+T_w*(-4.8796119E-7 + 1.0811263E-10*T_w)))
h_w = T_w*(1030.39 + T_w*(-2.6338079E-1/2+T_w*(7.5238345E-4/3+T_w*(-4.8796119E-7/4 + 1.0811263E-10/5*T_w))))

u_grad = 1.0 / R_nose * sqrt(2.0*(P_stag-P_inf)/rho_stag)
htc_stag = 0.704 *(rho_stag*mu_stag)**0.44 * (rho_w*mu_w)**0.06 * sqrt(u_grad) * (H_r - h_w)/ (T_r - T_w)
"""