"""
High Temperature Air Shock Solver

Assumptions:
- Local chemical and thermodynamic equilibrium

Applicable:
- Normal Shock
- Oblique Shock
- Conical Flow
- Prandtl-Meyer Expansion

- Make calls to functions in this file from a virtual environment (in conda) where cantera is installed.
- Both scalar and nparrays are accepted as input. Arrays are one dimensional.

Author:     Dogan Akcakaya
Date:       21.07.2023
"""

"""
Notes 1
Date:       25.07.2023

 - Since values are extracted from Cantera, vectorel multiplication without looping cannot be propely added.
 - Now both scalar and nparrays are accepted as input. Arrays are in (n,) shape.
 - Oblique shock solving requires some improvement for initial assumption and convergence path guiding.
 
Notes 2
Date:       31.07.2023

 - Stagnation condition is reached as adibatic and with that isentropic expansion. However, cantera solvers cannot
   solve equilibrium condition by constant entropy and enthalpy. Therefore a pressure ratio is assumed
   solved iteratively to equate initial and final entropy. Direction of iterations are changed based on
   errors.

 - Oblique shock still requires a iteration method. Method used in stagnation can be used there too.
 - Convergence is quite slow. Shock angle might not be appropriate for iteration.
 
 - 298 K is minimum temperature specified in gas model file (airNASA9.yaml). Can this temperature
   be modified without causing any problem in the model?
   
Notes 3
Date:       16.08.2023

 - cshock, conic shock (flow) solver is implemented.
 - Oblique shock initial guess is now polynomial relation (better). But iterative calculation
   is trigonometric. Fsolve is used with initial guess of previous beta to eliminate problems
   related to requirement of having different signs on the ends of range. Now, results might
   be larger than 90 degrees, which diverges iterations into a loop. Iterative change of 
   shock angle can be reduced by multipliers to not jump into that region.
   
   
 - Replace stagnation sf with si (entropy is constant, ignore error)

Notes 4

 - Normal shock and with that conic/oblique shock doesn't work well for very low Ma numbers.

Notes 5

 - Hard limits for conic and normal shocks added for low Ma numbers (1.15)
 - Perfect angle for conic is improved to yield consistent solutions (if the angle is complex, real component is extracted)
   which is actually reasonable since high shock angle 
 - A reliable method for convergence is implemented on conic shock solver. 

Notes 6
Date:       15.12.2023
 - Fixed oblique shock using previous versions
 - 
 
Notes 7
Date:       19.12.2023
 - Added newton raphson implementation for normal and oblique shock (results look good)
 - Numerical derivative a little bit risky (multiply with free parameters to scale, h * dratio)
 - Added newton raphson for stagnation (for %0.1 - %0.01 usually one iteration so not really necessary but anyways)
 
Notes 8
Date:       22.12.2023
 - Noticed that for vey low Ma numbers, results are being less accurate. So perfect gas solutions are added.
 - Scipy's newton raphson method can be used instead of self wrtitten NS method.
"""
# Dependicies
import cantera as ct
import numpy as np
import math
import sys, os
import matplotlib.pyplot as plt
import colorama
from colorama import Fore, Back, Style
import time

from numpy import *
from numpy import sqrt
from math import tan
from scipy.optimize import bisect, fsolve, newton
from scipy.integrate import solve_ivp

## Functions and current performance
# nshock:   solves normal shock. No problem encountered.
# oshock:   solves oblique shock. Bad guess of initial angle may lead to diverge.
# cshock:   solves conic shock. Bad guess of initial angle may lead to diverge. Also 0.05 
#           tolerance is added due to numerical problems encountered during solution.
# pmexpand: solves pm expansion. Requires improvements.
# eqsound:  works


colorama.init(autoreset=True)
"""
print(Fore.GREEN + 'High Temperature 1D Steady Gas Dynamics')
print('{0:>25}'.format('Running Cantera Version: ') + Fore.GREEN + ct.__version__,)
print('{0:>24}'.format('Last update:')  + Fore.GREEN + ' 15.12.2023', end = "\n\n")
"""
template = '{:<25} {:<14} {:<6}'

# functions can be executed in this class to supress their outputs
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# High Temperature Air Normal Shock
# Upstream parameters: u1, d1, p1, T1, h1
# Downstream parameters: u2, d2, p2, T2, h2

def nshock(M1, u1, d1, p1, T1, perfect='no', imax=200, errmax=0.01):
    
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1 = np.atleast_1d(M1, u1, d1, p1, T1)
    
    """
    Gas
    - Import high temperature gas to cantera
    - Set initial temperature, pressure and composition of air
    - Bring it to shock upstream conditions and equilibrate by minimizing gibbs free energy
    """
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Pre allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,))
    u2 = np.zeros(shape=(ndata,))
    d2 = np.zeros(shape=(ndata,))
    p2 = np.zeros(shape=(ndata,))
    T2 = np.zeros(shape=(ndata,))
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti = M1[i], u1[i], d1[i], p1[i], T1[i]
        
        print(Fore.YELLOW + "Normal Shock solution", i+1,"/",ndata)
        
        # get upstream conditions
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        hi = gas.h
        
        # printing templates // template 1 -> header / template 2 -> numerical outputs
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        print(template.format('Iteration', 'dratio', 'rel_err', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, hi, gas.s))
        
        # SUBSONIC CHECK
        if Mi <= 1:
            print(template2.format('SUBSONIC', '-', '-', Mi, '-', ui, di, pi, Ti, hi, gas.s))
            print(' -->', Fore.GREEN + 'Since the flow is subsonic/sonic no shock solution. Flow conditions are not changed.', end = '\n\n')
            
            M2[i], u2[i], d2[i], p2[i], T2[i] = M1[i], u1[i], d1[i], p1[i], T1[i]
            continue
            
        # PERFECT SOLUTION IF ASKED or T2 < 400K
        if perfect == 'yes' or (1+0.2*Mi**2)*(7*Mi**2-1)/(7.2*Mi**2) * Ti < 400: # Perfect gas solution
            M2[i] = np.sqrt(np.divide(0.4*(Mi**2) + 2, 2*1.4*(Mi**2)-0.4))
            d2[i] = di * np.divide(2.4*Mi**2, 2+0.4*Mi**2)
            p2[i] = pi * (2.8/2.4 * Mi**2 - 0.4/2.4)
            T2[i] = Ti * np.divide(p2[i]*di, pi*d2[i])
            u2[i] = ui * np.divide(di, d2[i])
            gas.TP = T2[i], p2[i]
            gas.equilibrate('TP')
            print(template2.format('PERFECT', '-', '-', M2[i], '-', u2[i], d2[i],  p2[i], T2[i], hi, gas.s))
            print(' -->', Fore.GREEN + "Perfect Gas Normal Shock solution")
            print('\n')
            continue
            
        """
        Iteration parameters
        - Initial guess of density ratio from perfect gas relations. (It was assumed 0.1 before)
        - imax is max number of iterations. Given in function definition.
        - errmax is max relative error percent. Given in function definition.
        """
        
        # function to find root (minimizing error)
        def f(dratio):
            pf = pi + di * (ui**2) * (1-dratio)
            hf = hi + (ui**2)/2 * (1-dratio**2)
            gas.HP = hf, pf
            gas.equilibrate('HP')
            df = gas.density
            Tf = gas.T
            err = (df - di/dratio)/(di/dratio) * 100
            return err, pf, hf, df, Tf
        
        dratio = (2.4*(Mi**2)/(2+0.4*Mi**2))**-1
        
        # Iterating to converge
        for j in range(1, imax+1):
            err = f(dratio)[0]
            
            pf = f(dratio)[1]
            hf = f(dratio)[2]
            df = f(dratio)[3]
            Tf = f(dratio)[4]
            
            uf = di * ui / df
            gas.HP = hf, pf
            af = eqsound(gas)
            Mf = uf/af
            
            template2 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.3f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template2.format(j, dratio, err, Mf, af, uf, df, pf, gas.T, hf, gas.s))
            
            if abs(err) < errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            h = 1e-6
            f_prime = (f(dratio+h)[0] - f(dratio)[0])/h
            # check if f_prime is too smaller
            
            dratio = dratio - err / f_prime
        print('\r'*(j+5))
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf
    return(M2, u2, d2, p2, T2)

def nshock_SCPY(M1, u1, d1, p1, T1, perfect='no', imax=200, errmax=0.01):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1 = np.atleast_1d(M1, u1, d1, p1, T1)
    
    """
    Gas
    - Import high temperature gas to cantera
    - Set initial temperature, pressure and composition of air
    - Bring it to shock upstream conditions and equilibrate by minimizing gibbs free energy
    """
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Pre allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,))
    u2 = np.zeros(shape=(ndata,))
    d2 = np.zeros(shape=(ndata,))
    p2 = np.zeros(shape=(ndata,))
    T2 = np.zeros(shape=(ndata,))
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti = M1[i], u1[i], d1[i], p1[i], T1[i]
        
        print(Fore.YELLOW + "Normal Shock solution", i+1,"/",ndata)
        
        # get upstream conditions
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        hi = gas.h
        
        # printing templates // template 1 -> header / template 2 -> numerical outputs
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        print(template.format('Iteration', 'dratio', 'rel_err', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, hi, gas.s))

        """
        Iteration parameters
        - Initial guess of density ratio from perfect gas relations. (It was assumed 0.1 before)
        - imax is max number of iterations. Given in function definition.
        - errmax is max relative error percent. Given in function definition.
        """
        
        # function to find root (minimizing error)
        def f(dratio):
            pf = pi + di * (ui**2) * (1-dratio)
            hf = hi + (ui**2)/2 * (1-dratio**2)
            gas.HP = hf, pf
            gas.equilibrate('HP')
            df = gas.density
            Tf = gas.T
            err = (df - di/dratio)/(di/dratio) * 100
            return err
        
        dratio = (2.4*(Mi**2)/(2+0.4*Mi**2))**-1
        
        dratio_new = newton(f, dratio)
        dratio = dratio_new 
        
        pf = pi + di * (ui**2) * (1-dratio)
        hf = hi + (ui**2)/2 * (1-dratio**2)
        gas.HP = hf, pf
        gas.equilibrate('HP')
        df = gas.density
        Tf = gas.T
        
        uf = di * ui / df
        af = eqsound(gas)
        Mf = uf/af
        
        print(Mf, uf, df, pf, Tf)
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf
    return(M2, u2, d2, p2, T2)

# High Temperature Air Oblique Shock solved using normal shock relations
# Upstream parameters: u1, d1, p1, T1, h1, theta
# Downstream parameters: u2, d2, p2, T2, h2, beta
# <theta: wedge angle>, <beta: wave angle>

def oshock(M1, u1, d1, p1, T1, theta, perfect='no', imax=20, errmax=0.01):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        print(Fore.YELLOW + "Oblique Shock Weak Solution", i+1,"/",ndata, ' - for theta =',thetai)
        
        thetai = math.radians(thetai) # work with radians
        
        # assume a wave angle (this needs improvement)
        # make an educated guess from perfect gas relations
        
        # perfect air angle solution
        def perfectair_angles(M, theta):
            k = 1.4
            A = M**2 - 1
            B = 0.5*(k+1)*M**4*np.tan(theta)
            C = (1+(k+1)/2*M**2)*np.tan(theta)
            coeffs = [1, C, -A, (B-A*C)]
            
            roots = np.array([r for r in np.roots(coeffs) if r > 0])
            
            betas = np.arctan(1/roots)
            beta_weak = np.min(betas)
            beta_strong = np.max(betas)
            return(beta_weak, beta_strong)
            
        beta, _ = perfectair_angles(Mi, thetai)
        beta = beta.real    # DISCUSS THIS ****
        # alternative, not working properly
        """
        def initial_beta(beta):
            return ((1.2*(Mi**2)/(Mi**2*math.sin(beta)**2-1)-1)*tan(beta)*tan(thetai)-1)
        # beta = 1.2 * thetai
        # beta = bisect(initial_beta, thetai, 1.7) # 1.7 is selected arbitrary
        # beta = math.radians(15 + 5)
        # beta = 17.4*math.pi/180
        # beta = fsolve(initial_beta, thetai) # alternative root finder
        # beta = beta[0]
        """
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Wave Angle B', 'rel_err', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        itdir = 1
        err_old = 0
        # iterate through wave angles
        for j in range(1,imax+1):
            # components of velocity
            uni, ut = ui * math.sin(beta), ui * math.cos(beta)
            Mni, Mti = Mi * math.sin(beta), Mi * math.cos(beta)

            # relation solution of downstream normal component
            # unf_rel = uni * math.tan(beta-thetai) / math.tan(beta)
            
            with HiddenPrints():
                # normal shock solution of downstream normal component
                Mnf, unf, _, pf, Tf = nshock(Mni, uni, d1, p1, T1)
                Mnf, unf = Mnf.item(), unf.item()
                gas.TP = Tf[0], pf[0]
                gas.equilibrate('TP')
            
            # calculate beta angle from normal component ratios
            def oangle(betaf):
                return (np.tan(betaf-thetai) - unf / uni * np.tan(betaf))
            # betaf = bisect(oangle, thetai, 1.7)
            
            betaf = ndarray.item(fsolve(oangle, beta))

            uf = sqrt(unf**2 + ut**2)
            af = eqsound(gas)
            Mf = uf/af

            err = (betaf - beta)/beta * 100
            # print(beta*180/math.pi, betaf*180/math.pi)
            
            template2 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.3f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template2.format(j, beta*180/math.pi, err, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif abs(err) > errmax and j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
                
            if j != 1:
                if abs(err) > abs(err_old):
                    if itdir == 1:
                        itdir = 0
                    else:
                        itdir = 1
            # iteration direction
            if itdir == 1:
                beta = beta - 0.05 * beta * err / 10
            else:
                beta = beta + 0.05 * beta * err / 10
                
            err_old = err
        """
        # Iterating to converge
        for j in range(1, imax+1):
            pf = pi + di * (ui**2) * (1-dratio)
            hf = hi + (ui**2)/2 * (1-dratio**2)
            
            gas.HP = hf, pf
            gas.equilibrate('HP')
            df = gas.density
            
            err = (df - di/dratio)/di * 100
            
            print('Iteration: ', j, " dratio: ", "%0.3f" % dratio, " relative_error: ", "%0.3f" % err)
            print('u2:', di*ui/df, 'd2:', df, 'p2:', pf, 'T2:', gas.T, 'h2:', hf, end = '\n\n')
            if abs(err) <= errmax:
                print ('Converged in ', j, ' iterations', end = "\n\n") 
                break
            dratio = di / df;
        
        uf = di * ui / df
        Tf = gas.T
        Rf = gas.P / gas.density / gas.T
        Mf = uf / sqrt(gas.cp/gas.cv * Rf * gas.T)
        
        M2[i][0] = Mf
        u2[i][0] = uf
        d2[i][0] = df
        p2[i][0] = pf
        T2[i][0] = Tf
        """
        print('\n')
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, gas.density, gas.P, gas.T
    return(M2, u2, d2, p2, T2, beta)

# NEWTON RAPHSON IMPLEMENTED oSHOCK
def oshock_NRS(M1, u1, d1, p1, T1, theta, perfect='no', imax=20, errmax=0.01):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        print(Fore.YELLOW + "Oblique Shock Weak Solution", i+1,"/",ndata, ' - for theta =',thetai)
        
        thetai = math.radians(thetai) # work with radians
        
        # assume a wave angle (this needs improvement)
        # make an educated guess from perfect gas relations
        
        # perfect air angle solution
        def perfectair_angles(M, theta):
            k = 1.4
            A = M**2 - 1
            B = 0.5*(k+1)*M**4*np.tan(theta)
            C = (1+(k+1)/2*M**2)*np.tan(theta)
            coeffs = [1, C, -A, (B-A*C)]
            
            roots = np.array([r for r in np.roots(coeffs) if r > 0])
            
            betas = np.arctan(1/roots)
            beta_weak = np.min(betas)
            beta_strong = np.max(betas)
            return(beta_weak, beta_strong)
            
        beta, beta_strong = perfectair_angles(Mi, thetai)
        beta = beta.real    # DISCUSS THIS ****
        # NOTE, INTIAL BETA VALUE DETERMINES WHETHER SOLUTION IS DRIVEN TO STRONG OR WEAK SOLUTION.
        # USE SQRT(BETA_REAL^2 + BETA_IM^2) AS STRONG INITIAL, AND SQRT(BETA_REAL^2 - BETA_IM^2)
        # OR (BETA_REAL + BETA_IM) AND (BETA_REAL - BETA_IM)
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Wave Angle B', 'rel_err', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        def f(beta):
            # components of velocity
            uni, ut = ui * math.sin(beta), ui * math.cos(beta)
            Mni, Mti = Mi * math.sin(beta), Mi * math.cos(beta)
            with HiddenPrints():
                Mnf, unf, _, pf, Tf = nshock(Mni, uni, d1, p1, T1)
                Mnf, unf = Mnf.item(), unf.item()
                gas.TP = Tf[0], pf[0]
                gas.equilibrate('TP')
            
            # calculate beta angle from normal component ratios
            def oangle(betaf):
                return (np.tan(betaf-thetai) - unf / uni * np.tan(betaf))
            # betaf = bisect(oangle, thetai, 1.7)
            betaf = ndarray.item(fsolve(oangle, beta))

            uf = sqrt(unf**2 + ut**2)
            af = eqsound(gas)
            Mf = uf/af
            
            err = (betaf - beta)/beta * 100
            return err, uf, af, Mf
            
        # iterate through wave angles
        for j in range(1,imax+1):
            err = f(beta)[0]
            
            uf = f(beta)[1]
            af = f(beta)[2]
            Mf = f(beta)[3]
            
            template2 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.3f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template2.format(j, beta*180/math.pi, err, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            h = 1e-6
            f_prime = (f(beta+h)[0] - f(beta)[0])/h
            # check if f_prime is too smaller
            
            beta = beta - err / f_prime
            
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, gas.density, gas.P, gas.T
        print('\n')
    return(M2, u2, d2, p2, T2, beta)
# High Temperature Air Isentropic expansion to stagnation conditions
# Upstream parameters: u1, d1, p1, T1, h1
# Downstream parameters: d2, p2, T2, h2

def stagnation(M1, u1, d1, p1, T1, perfect='no', imax=100, errmax=0.01):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1 = np.atleast_1d(M1, u1, d1, p1, T1)
    
    """
    Gas
    - Import high temperature gas to cantera
    - Set initial temperature, pressure and composition of air
    """ 

    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')  
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Pre allocate output arrays
    ndata = M1.shape[0]
    d2 = np.zeros(shape=(ndata,))
    p2 = np.zeros(shape=(ndata,))
    T2 = np.zeros(shape=(ndata,))
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti = M1[i], u1[i], d1[i], p1[i], T1[i]
        
        print(Fore.YELLOW+ "Stagnation ", i+1,"/",ndata)
         
        # get upstream conditions
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        hi = gas.h
        si = gas.s
        
        # downstream conditions
        hf = hi + ui**2 / 2
        sf = si
        
        """
        Iteration parameters
        - Initial guess of pressure ratio is made from isentropic perfect gas relation for k = 1.4.
        - imax is max number of iterations. Given in function definition.
        - errmax is max relative error percent. Given in function definition.
        """       
        
        pratio = (1 + 0.2 * Mi**2)**-3.5
        itdir = 1 # iteration direction
        err_old = 0

        # Iterating to converge
        for j in range(1, imax+1):
            pf = pi/pratio
            gas.HP = hf, pf
            gas.equilibrate('HP')
            
            sf = gas.s
            Tf = gas.T
            df = gas.density

            err = (sf - si)/si * 100
            
            print('->', 'Iteration: ', j, " pratio: ", "%0.3f" % pratio, " relative_error: ", "%0.3f" % err)
            print('->', 'd2:', "%0.3f" % df, 'p2:', "%0.3f" % pf, 'T2:', "%0.3f" % Tf, 'h2:', "%0.3f" % hf, 's2: ', "%0.3f" % sf, end = '\n\n')
            
            # convergence criterion
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif abs(err) > errmax and j == imax:
                print('-->', Fore.RED + 'Could not converge. Stopped.')
                exit()
                
            if j != 1:
                if abs(err) > abs(err_old):
                    if itdir == 1:
                        itdir = 0
                    else:
                        itdir = 1
                        
            # iteration direction
            if itdir == 1:
                pratio = pratio - 2 * pratio * err / 100
            else:
                pratio = pratio + 2 * pratio * err / 100
                            
            err_old = err
            print('-' * 50, end = '\n')
            """
            older method for iteration direction
            not working properly
            
                if hf > 0:
                    if err > 0: # Entropy is higher. Stagnation pressure should be increased to reduce entropy.
                        pratio = pratio + pratio * err / 100
                    else: # err < 0 Entropy is smaller. 
                        pratio = pratio - pratio * err / 100
                else:
                    if err > 0: # Entropy is higher. Stagnation pressure should be increased to reduce entropy.
                        pratio = pratio + pratio * err / 100
                    else: # err < 0 Entropy is smaller. 
                        pratio = pratio - pratio * err / 100
            """
            
        print('-' * 50, end = '\n\n')
        d2[i] = df
        p2[i] = pf
        T2[i] = Tf
    return(d2, p2, T2)

def stagnation_NRS(M1, u1, d1, p1, T1, perfect='no', imax=100, errmax=0.01):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1 = np.atleast_1d(M1, u1, d1, p1, T1)
    
    """
    Gas
    - Import high temperature gas to cantera
    - Set initial temperature, pressure and composition of air
    """ 

    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')  
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Pre allocate output arrays
    ndata = M1.shape[0]
    d2 = np.zeros(shape=(ndata,))
    p2 = np.zeros(shape=(ndata,))
    T2 = np.zeros(shape=(ndata,))
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti = M1[i], u1[i], d1[i], p1[i], T1[i]
        
        print(Fore.YELLOW+ "Stagnation ", i+1,"/",ndata)
         
        # get upstream conditions
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        hi = gas.h
        si = gas.s
        
        # downstream conditions
        hf = hi + ui**2 / 2
        sf = si
        
        """
        Iteration parameters
        - Initial guess of pressure ratio is made from isentropic perfect gas relation for k = 1.4.
        - imax is max number of iterations. Given in function definition.
        - errmax is max relative error percent. Given in function definition.
        """       
        
        def f(pratio):
            pf = pi/pratio
            gas.HP = hf, pf
            gas.equilibrate('HP')
        
            sf = gas.s
            Tf = gas.T
            df = gas.density
            err = (sf - si)/si * 100
            
            return err, pf, sf, Tf, df
        
        pratio = (1 + 0.2 * Mi**2)**-3.5
        
        # Iterating to converge
        for j in range(1, imax+1):
            err = f(pratio)[0]
            
            pf = f(pratio)[1]
            sf = f(pratio)[2]
            Tf = f(pratio)[3]
            df = f(pratio)[4]
            
            print('->', 'Iteration: ', j, " pratio: ", "%0.3f" % pratio, " relative_error: ", "%0.3f" % err)
            print('->', 'd2:', "%0.3f" % df, 'p2:', "%0.3f" % pf, 'T2:', "%0.3f" % Tf, 'h2:', "%0.3f" % hf, 's2: ', "%0.3f" % sf, end = '\n\n')
            
            # convergence criterion
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif j == imax:
                print('-->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            h = 1e-6
            f_prime = (f(pratio+h)[0] - f(pratio)[0])/h
            
            pratio = pratio - err / f_prime
            
        print('-' * 50, end = '\n\n')
        d2[i], p2[i], T2[i] = df, pf, Tf
    return(d2, p2, T2)
# Equilibrium speed of sound
# Calculates speed of sound by perturbating pressure
def eqsound(gas, rtol=1.0e-6, max_iter=5000):
    ## equilibrate
    gas.equilibrate('TP', rtol=rtol, max_iter=max_iter)
    
    ## chemical equilibrium state
    s0 = gas.s
    p0 = gas.P
    rho0 = gas.density
    
    ## perturb pressure
    p1 = p0*1.0001
    
    ## new state
    gas.SP = s0, p1
    
    ## chemical equilibruim
    gas.equilibrate('SP', rtol=rtol, max_iter=max_iter)
    soundequil = math.sqrt((p1-p0)/(gas.density-rho0))
    
    return soundequil

# Perfect gas Normal shock + stagnation for comparison
# might be wrong need to check
def pnshockstag(M1, u1, d1, p1, T1):
    # convert scalars into at least 1d array
    M1 = np.atleast_1d(M1, u1, d1, p1, T1)
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Pre allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,))
    u2 = np.zeros(shape=(ndata,))
    d2 = np.zeros(shape=(ndata,))
    p2 = np.zeros(shape=(ndata,))
    T2 = np.zeros(shape=(ndata,))
    
    print("Perfect Gas Normal Shock solution")
    M2 = np.sqrt(np.divide(0.4*(M1**2) + 2, 2*1.4*(M1**2)-0.4))
    d2 = d1 * np.divide(2.4*M1**2, 2+0.4*M1**2)
    p2 = p1 * (2.8/2.4 * M1**2 - 0.4/2.4)
    T2 = T1 * np.divide(p2*d1, p1*d2)
    u2 = u1 * np.divide(p1, p2)
    T0 = T2 * (1 + 0.2 * M2**2)
    print(M2, T2/T1, T0)
    return(M1, T0)

# High Temperature Conic Shock
# Three dimensional (relieved), axisymmetric, radial gradients are zero (conic flow)
def cshock(M1, u1, d1, p1, T1, theta, perfect='no', imax=200, errmax=1):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        
        print(Fore.YELLOW + "Conic Shock Weak Solution", i+1,"/",ndata, ' - for theta =',thetai, Style.BRIGHT + 'at cone surface')
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Wave Angle B', 'Vt (error)', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        # For very low supersonic numbers, numerical values are hard to get. Therefore a tolerance is added for no shock condition.
        if Mi <= 1 + 0.15:
            print(template2.format('SUBSONIC', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
            print(' -->', Fore.GREEN + 'Since the flow is subsonic/sonic no shock solution. Flow conditions are not changed.', end = '\n\n')
            
            M2[i], u2[i], d2[i], p2[i], T2[i] = Mi, ui, di, pi, Ti
            continue
        
        thetai = math.radians(thetai) # work with radians
        
        # use approximate and ideal solution for low mach solutions (graphical method)
        if Mi <= 1:
            SM_5_1 = array([1.08, 1.54, 2.08, 2.59, 3.21, 4.12, 4.70, 5.49, 5.71, 5.95])
            SM_5_2 = array([1.03, 1.50, 2.01, 2.50, 3.07, 3.92, 4.44, 5.14, 5.32, 5.53])
            SM_10_1 = array([5.94, 5.32, 4.98, 4.42, 4.05, 3.69, 3.39, 3.04, 2.80, 2.50, 2.08, 1.74, 1.34, 1.13, 1.07])
            SM_10_2 = array([4.94, 4.51, 4.26, 3.85, 3.57, 3.30, 3.05, 2.75, 2.55, 2.28, 1.92, 1.62, 1.24, 1.01, 0.93])
            SM_15_1 = array([5.97, 5.56, 5.21, 4.87, 4.40, 3.98, 3.57, 3.24, 2.85, 2.49, 2.19, 1.83, 1.51, 1.26, 1.12])
            SM_15_2 = array([4.35, 4.14, 3.95, 3.76, 3.48, 3.20, 2.92, 2.68, 2.39, 2.11, 1.87, 1.57, 1.29, 1.04, 0.86])

            PR_5_1 = array([5.95, 5.39,  4.80, 4.11, 3.36, 2.79, 2.36, 1.80, 1.09])
            PR_5_2 = array([1.53, 1.47,  1.39, 1.29, 1.21, 1.16, 1.12, 1.09, 1.05])
            PR_10_1 = array([5.98, 5.42, 5.03 ,4.47, 3.95, 3.48, 2.91, 2.28, 1.76, 1.37, 1.09])
            PR_10_2 = array([2.78, 2.50, 2.31 ,2.06, 1.86, 1.69, 1.50, 1.33, 1.23, 1.17, 1.14])
            PR_15_1 = array([5.98, 5.75, 5.51 ,5.22, 4.93, 4.72, 4.36, 4.00, 3.55, 3.02, 2.43, 2.10, 1.68, 1.45, 1.16])
            PR_15_2 = array([4.76, 4.50, 4.26 ,3.94, 3.65, 3.43, 3.12, 2.81, 2.45, 2.11, 1.75, 1.59, 1.41, 1.37, 1.34])

            temp_ratio, int_fac, f1, f2, f3, f4 = 0., 0., 0., 0., 0., 0.
            if (thetai >= 0.087266 and  thetai <= 0.1745329):            
                int_fac = (thetai - 0.087266) / 0.087266
                f1 = interp(Mi, SM_5_1, SM_5_2)
                f2 = interp(Mi, SM_10_1, SM_10_2)
                f3 = interp(Mi, PR_5_1, PR_5_2)
                f4 = interp(Mi, PR_10_1, PR_10_2)
            elif (thetai > 0.1745329 and thetai <= 0.2617994):
                int_fac = (thetai - 0.1745329) / 0.087266
                f1 = interp(Mi, SM_10_1, SM_10_2)
                f2 = interp(Mi, SM_15_1, SM_15_2)
                f3 = interp(Mi, PR_10_1, PR_10_2)
                f4 = interp(Mi, PR_15_1, PR_15_2)
                
            M2[i] = f1 + int_fac * (f2-f1)
            p2[i] = p1[i] * (f3 + int_fac * (f4-f3))
            
            temp_ratio = (1.+0.2*Mi**2) / (1.+0.2*M2[i]**2)
            T2[i] = Ti * temp_ratio
            u2[i] = M2[i] * sqrt(401.856*T2[i])
            d2[i] = di * temp_ratio**2.5
            try:
                gas.TP = T2[i], p2[i]
                gas.equilibrate('TP')
                print(template2.format('GR-IDEAL', '-', '-', M2[i], '-', u2[i], d2[i], p2[i], T2[i], gas.h, gas.s))
            except Exception as inst:
                print(Fore.RED + ' Conic shock solution failed while using graphical method.'.center(169))
                exit()
            continue

        # Cone angle is smaller than oblique shock angle due to three dimensionality
        # For start point, perfect air oblique shock angle can be assumed
        
        # make an educated guess from perfect gas relations
        # perfect air angle solution using polynomial relation instead of trigonometric relation (latter is harder to solve numerically)
        
        def perfectair_angles(M, theta):
            k = 1.4
            A = M**2 - 1
            B = 0.5*(k+1)*M**4*np.tan(theta)
            C = (1+(k+1)/2*M**2)*np.tan(theta)
            coeffs = [1, C, -A, (B-A*C)]
            
            roots = np.array([r for r in np.roots(coeffs) if r > 0])
            
            betas = np.arctan(1/roots)
            beta_weak = np.real(np.min(betas))
            beta_strong = np.real(np.max(betas))
            return(beta_weak, beta_strong)
        
        beta, beta_strong = perfectair_angles(Mi, thetai)
        # print(beta*180/math.pi, beta_strong*180/math.pi)

        # cancelled
        # noticed that real parts of shock angles (if they are complex) are actual solutions.
        """
        if np.iscomplex(beta) == True_:
            beta = thetai * 1.01
        """
        
        # alternative trigonometric method
        # not working properly for every range, needs adjustment
        
        """
        def initial_beta(beta):
            return ((1.2*(Mi**2)/(Mi**2*math.sin(beta)**2-1)-1)*tan(beta)*tan(thetai)-1)
        # beta = 1.2 * thetai
        # beta = bisect(initial_beta, thetai, 1.7) # 1.7 is selected arbitrary
        # beta = math.radians(15 + 5)
        # beta = 17.4*math.pi/180
        # beta = fsolve(initial_beta, thetai) # alternative root finder
        # beta = beta[0]
        """

        min_beta = 0
        err_old = 0
        max_beta = 0
        beta_old = 0
        beta_fix_step = 0.5 * math.pi/180
        beta = 78.6*math.pi/180
        # iterate through solution
        j = 1
        while j < imax+1:
            j = j + 1
            # components of velocity
            uni, ut = ui * math.sin(beta), ui * math.cos(beta)
            Mni, Mti = Mi * math.sin(beta), Mi * math.cos(beta)
            
            """
            print(beta*180/math.pi)
            if Mni <= 1 + 0.01:
                print(' -->', Fore.GREEN + 'Normal component is subsonic/sonic. Flow conditions are not changed.', end = '\n\n')
                Mf, uf, df, pf, Tf = M1[i], u1[i], d1[i], p1[i], T1[i]
                continue
            """
            # print(beta)
            # just normal component just after the shock using nshock
            with HiddenPrints():
                # normal shock solution of downstream normal component
                Mnf, unf, _, pf, Tf = nshock(Mni, uni, d1, p1, T1)
                Mnf, unf, pf, Tf = Mnf.item(), unf.item(), pf.item(), Tf.item()
                gas.TP = Tf, pf
                gas.equilibrate('TP')
            
            uf = sqrt(unf**2 + ut**2)
            af = eqsound(gas)
            Mf = uf/af
            sf = gas.s # constant

            # now the shock surface conditions are defined
            # the flowfield between shock and cone is isentropic
            # but unlike the wedge, the flow is three dimensional, the streamlines are curved.
            # this region is bounded by three coupled first order ODE (conservation) and two thermodynamic relations (chemical equilibrium)
            
            # define system of ODEs
            # variables: Vt, Vr, p
            # state variables: p, sf
            # <t: range, V: initial conditions>

            def conic_ODEs(t, V):
                Vr, Vt, p = V # initial conditions
                
                gas.SP = sf, p # set state
                gas.equilibrate('SP') # set state

                a = eqsound(gas) # speed of sound
                rho = gas.density # density
                
                dVr_dt = Vt # (1)
                dVt_dt = a**2 / (Vt**2 - a**2) * (2*Vr + Vt / math.tan(t) - Vr * Vt**2 / a**2) # (2)
                dp_dt = - rho * Vt * a**2 / (Vt**2 - a**2) * (Vr + Vt / math.tan(t)) # (3)
                ddt = [dVr_dt, dVt_dt, dp_dt]
                return ddt
            
            # [beta, thetai]
            t = np.linspace(beta, thetai, 50)
            
            # Note for right hand coordinate system, Vt should be negative.
            # ut is tangential velocicity after oblique shock for initial condition, 
            # not related to theta in system of ODEs.
            
            try:
                sol = solve_ivp(conic_ODEs, [beta, thetai], [ut, -unf, pf], t_eval = t, dense_output='true')
            except Exception as inst:
                # print(inst)
                print(Fore.RED + ' Conic shock solution failed for this iteration.'.center(169))
                if j-1 == imax:
                    print(' -->', Fore.RED + 'Could not converge. Stopped.')
                    exit()
                # perfect oblique shock angle is way higher than conic shock angle in some cases
                # therefore, update reduce the angle by a percent.
                beta = 0.95*beta
                continue
            
            Vt = sol.y[1][-1] # cone surface Vt
            Vr = sol.y[0][-1] # cone surface Vr
            p = sol.y[2][-1] # cone surface p
            
            # Vt should be zero.
            err = Vt
            
            current_sign = 1 if err > 0 else -1
            
            # set surface state
            gas.SP = sf, p
            pf = p
            Tf = gas.T
            
            af = eqsound(gas)
            uf = sqrt(Vt**2 + Vr**2)
            Mf = uf / af
            
            df = gas.density
            
            template3 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.6f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template3.format(j-1, beta*180/math.pi, Vt, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            ### CONVERGENCE CHECK, STEP SIZE, CONVERGENCE ACCEELERATION/DECELERATION CHECKS ARE MADE BELOW
            ### REQUIRES HEAVY IMPROVEMENTS ITERATION HANDLING
            
            ### WEDGE ANGLE IS HIGHER THAN CONIC SHOCK ANGLE. SO ITERATIONS ARE DIRECTED TOWARDS THE WAY
            ### TO REDUCE ANGLE. ALSO IF ANGLE IS LARGER THAN ACTUAL ANGLE, Vt (ERROR) IS POSITIVE.
            
            ### EQUATIONS ARE MODETERALY NON-LINEAR SO CLASSICAL ITERATION METHODS TEND TO FAIL OR OVERSHOOT
            ### FOR A SAFE GUIDING, A RANGE WITH OPPOSITE ENDS CONTAINING THE SOLUTION IS CAPTURED
            ### THEN IT IS TIGHTENED TO GET A APPROPRITE SOLUTION. FOR SOME CASES THE PROCESS MIGHT GET
            ### INTO A LOOP. IN THESE CASES, REDUCE THE STEP SIZE IN 'NO OPPOSITE SIGN' CASE. 
            
            ### NEWTON RAPHSON METHOD CAN BE IMPLEMENTED HOWEVER, IN SOME CASES AT THE BEGINNING OF THE ITERATIONS
            ### IDEAL OBLIQUE SHOCK ANGLE YIELDS 'BAD' THERMODYNAMIC VALUES TO BREAK CANTERA. IN THESE CONDITIONS
            ### BETA ANGLE IS REDUCED BY PERCENT UNTIL A THERMODYNAMICALLY APPROPRIATE SOLUTION IS OBTAINED.
            ### THIS MODIFICATION IS INTEGRAL PART OF ITERATION LOOP, SO NEWTON RAPHSON HAVE TO BE IMPLEMENTED
            ### MANUALLY. (DON'T USE EXTERNAL METHODS)
            
            # convergence check
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % (j-1), end = "\n")
                break
            elif abs(err) > errmax and j == imax:
                if imax >= 1000:
                    print(' -->', Fore.RED + 'Could not converge. Stopped.')
                    exit()
                else:
                    imax = imax + 200
                    beta_fix_step = beta_fix_step * 0.5
                    continue
            
            # find a range with opposite signs for solution
            # sign check
            
            if 'previous_sign' in locals():
                if current_sign != previous_sign:
                    # opposite signs found, update beta range
                    if current_sign == 1:
                        max_beta = beta
                        beta = (max_beta*abs(err) + min_beta*abs(err_old))/(abs(err)+abs(err_old))
                    else:
                        min_beta = beta
                        beta = (max_beta*abs(err_old) + min_beta*abs(err))/(abs(err)+abs(err_old))
                    # beta = (max_beta + min_beta)/2
                else:
                    # no opposite signs, use regular step
                    if current_sign == 1:
                        """
                        if abs(err) > abs(err_old):
                            max_beta = beta_old
                            min_beta = beta
                            beta = (max_beta + min_beta)/2
                        else:
                            beta = beta - beta_fix_step*abs(err)*0.1
                        """
                        max_beta = beta
                        beta = beta - beta_fix_step
                    else:
                        """
                        if abs(err) > abs(err_old):
                            max_beta = beta_old
                            min_beta = beta
                            beta = (max_beta + min_beta)/2
                        else:
                            beta = beta + beta_fix_step*abs(err)*0.1
                        """
                        min_beta = beta
                        beta = beta + beta_fix_step
            else:
                # first iteration
                if current_sign == 1:
                    max_beta = beta
                    beta = beta - beta_fix_step
                else:
                    min_beta = beta
                    beta = beta + beta_fix_step
            # print(min_beta*180/math.pi, beta*180/math.pi, max_beta*180/math.pi)
            err_old = err
            beta_old = beta
            previous_sign = current_sign
        print('\n')
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf
    return(M2, u2, d2, p2, T2)

def cshock_NRS(M1, u1, d1, p1, T1, theta, perfect='no', imax=200, errmax=1):
    # convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    # Initialize gas
    # It would be better to initialize gas outside of functions and pass gas object to the function
    gas = ct.Solution('airNASA9-transport.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        
        print(Fore.YELLOW + "Conic Shock Weak Solution", i+1,"/",ndata, ' - for theta =',thetai, Style.BRIGHT + 'at cone surface')
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Wave Angle B', 'Vt (error)', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        # For very low supersonic numbers, numerical values are hard to get. Therefore a tolerance is added for no shock condition.
        if Mi <= 1 + 0.15:
            print(template2.format('SUBSONIC', '-', '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
            print(' -->', Fore.GREEN + 'Since the flow is subsonic/sonic no shock solution. Flow conditions are not changed.', end = '\n\n')
            
            M2[i], u2[i], d2[i], p2[i], T2[i] = Mi, ui, di, pi, Ti
            continue
        
        thetai = math.radians(thetai) # work with radians
        
        # use approximate and ideal solution for low mach solutions (graphical method)
        if Mi <= 1:
            SM_5_1 = array([1.08, 1.54, 2.08, 2.59, 3.21, 4.12, 4.70, 5.49, 5.71, 5.95])
            SM_5_2 = array([1.03, 1.50, 2.01, 2.50, 3.07, 3.92, 4.44, 5.14, 5.32, 5.53])
            SM_10_1 = array([5.94, 5.32, 4.98, 4.42, 4.05, 3.69, 3.39, 3.04, 2.80, 2.50, 2.08, 1.74, 1.34, 1.13, 1.07])
            SM_10_2 = array([4.94, 4.51, 4.26, 3.85, 3.57, 3.30, 3.05, 2.75, 2.55, 2.28, 1.92, 1.62, 1.24, 1.01, 0.93])
            SM_15_1 = array([5.97, 5.56, 5.21, 4.87, 4.40, 3.98, 3.57, 3.24, 2.85, 2.49, 2.19, 1.83, 1.51, 1.26, 1.12])
            SM_15_2 = array([4.35, 4.14, 3.95, 3.76, 3.48, 3.20, 2.92, 2.68, 2.39, 2.11, 1.87, 1.57, 1.29, 1.04, 0.86])

            PR_5_1 = array([5.95, 5.39,  4.80, 4.11, 3.36, 2.79, 2.36, 1.80, 1.09])
            PR_5_2 = array([1.53, 1.47,  1.39, 1.29, 1.21, 1.16, 1.12, 1.09, 1.05])
            PR_10_1 = array([5.98, 5.42, 5.03 ,4.47, 3.95, 3.48, 2.91, 2.28, 1.76, 1.37, 1.09])
            PR_10_2 = array([2.78, 2.50, 2.31 ,2.06, 1.86, 1.69, 1.50, 1.33, 1.23, 1.17, 1.14])
            PR_15_1 = array([5.98, 5.75, 5.51 ,5.22, 4.93, 4.72, 4.36, 4.00, 3.55, 3.02, 2.43, 2.10, 1.68, 1.45, 1.16])
            PR_15_2 = array([4.76, 4.50, 4.26 ,3.94, 3.65, 3.43, 3.12, 2.81, 2.45, 2.11, 1.75, 1.59, 1.41, 1.37, 1.34])

            temp_ratio, int_fac, f1, f2, f3, f4 = 0., 0., 0., 0., 0., 0.
            if (thetai >= 0.087266 and  thetai <= 0.1745329):            
                int_fac = (thetai - 0.087266) / 0.087266
                f1 = interp(Mi, SM_5_1, SM_5_2)
                f2 = interp(Mi, SM_10_1, SM_10_2)
                f3 = interp(Mi, PR_5_1, PR_5_2)
                f4 = interp(Mi, PR_10_1, PR_10_2)
            elif (thetai > 0.1745329 and thetai <= 0.2617994):
                int_fac = (thetai - 0.1745329) / 0.087266
                f1 = interp(Mi, SM_10_1, SM_10_2)
                f2 = interp(Mi, SM_15_1, SM_15_2)
                f3 = interp(Mi, PR_10_1, PR_10_2)
                f4 = interp(Mi, PR_15_1, PR_15_2)
                
            M2[i] = f1 + int_fac * (f2-f1)
            p2[i] = p1[i] * (f3 + int_fac * (f4-f3))
            
            temp_ratio = (1.+0.2*Mi**2) / (1.+0.2*M2[i]**2)
            T2[i] = Ti * temp_ratio
            u2[i] = M2[i] * sqrt(401.856*T2[i])
            d2[i] = di * temp_ratio**2.5
            try:
                gas.TP = T2[i], p2[i]
                gas.equilibrate('TP')
                print(template2.format('GR-IDEAL', '-', '-', M2[i], '-', u2[i], d2[i], p2[i], T2[i], gas.h, gas.s))
            except Exception as inst:
                print(Fore.RED + ' Conic shock solution failed while using graphical method.'.center(169))
                exit()
            continue

        # Cone angle is smaller than oblique shock angle due to three dimensionality
        # For start point, perfect air oblique shock angle can be assumed
        
        # make an educated guess from perfect gas relations
        # perfect air angle solution using polynomial relation instead of trigonometric relation (latter is harder to solve numerically)
        
        def perfectair_angles(M, theta):
            k = 1.4
            A = M**2 - 1
            B = 0.5*(k+1)*M**4*np.tan(theta)
            C = (1+(k+1)/2*M**2)*np.tan(theta)
            coeffs = [1, C, -A, (B-A*C)]
            
            roots = np.array([r for r in np.roots(coeffs) if r > 0])
            
            betas = np.arctan(1/roots)
            beta_weak = np.real(np.min(betas))
            beta_strong = np.real(np.max(betas))
            return(beta_weak, beta_strong)
        
        beta, beta_strong = perfectair_angles(Mi, thetai)
        
        def f(beta):
            uni, ut = ui * math.sin(beta), ui * math.cos(beta)
            Mni, Mti = Mi * math.sin(beta), Mi * math.cos(beta)
            
            with HiddenPrints():
                # normal shock solution of downstream normal component
                Mnf, unf, _, pf, Tf = nshock(Mni, uni, d1, p1, T1)
                Mnf, unf, pf, Tf = Mnf.item(), unf.item(), pf.item(), Tf.item()
                gas.TP = Tf, pf
                gas.equilibrate('TP')
            
            uf = sqrt(unf**2 + ut**2)
            af = eqsound(gas)
            Mf = uf/af
            sf = gas.s # constant
            
            def conic_ODEs(t, V):
                Vr, Vt, p = V # initial conditions
                    
                gas.SP = sf, p # set state
                gas.equilibrate('SP') # set state

                a = eqsound(gas) # speed of sound
                rho = gas.density # density
                    
                dVr_dt = Vt # (1)
                dVt_dt = a**2 / (Vt**2 - a**2) * (2*Vr + Vt / math.tan(t) - Vr * Vt**2 / a**2) # (2)
                dp_dt = - rho * Vt * a**2 / (Vt**2 - a**2) * (Vr + Vt / math.tan(t)) # (3)
                ddt = [dVr_dt, dVt_dt, dp_dt]
                return ddt
            
            # [beta, thetai]
            t = np.linspace(beta, thetai, 50)
            
            sol = solve_ivp(conic_ODEs, [beta, thetai], [ut, -unf, pf], t_eval = t, dense_output='true')
            
            Vt = sol.y[1][-1] # cone surface Vt
            Vr = sol.y[0][-1] # cone surface Vr
            p = sol.y[2][-1] # cone surface p
            
            # Vt should be zero.
            err = Vt
            
            return err
        
        for j in range(1,imax+1):
            err = f(beta)
        
            uni, ut = ui * math.sin(beta), ui * math.cos(beta)
            Mni, Mti = Mi * math.sin(beta), Mi * math.cos(beta)
            
            with HiddenPrints():
                # normal shock solution of downstream normal component
                Mnf, unf, _, pf, Tf = nshock(Mni, uni, d1, p1, T1)
                Mnf, unf, pf, Tf = Mnf.item(), unf.item(), pf.item(), Tf.item()
                gas.TP = Tf, pf
                gas.equilibrate('TP')
            
            uf = sqrt(unf**2 + ut**2)
            af = eqsound(gas)
            Mf = uf/af
            sf = gas.s # constant
            
            def conic_ODEs(t, V):
                Vr, Vt, p = V # initial conditions
                    
                gas.SP = sf, p # set state
                gas.equilibrate('SP') # set state

                a = eqsound(gas) # speed of sound
                rho = gas.density # density
                    
                dVr_dt = Vt # (1)
                dVt_dt = a**2 / (Vt**2 - a**2) * (2*Vr + Vt / math.tan(t) - Vr * Vt**2 / a**2) # (2)
                dp_dt = - rho * Vt * a**2 / (Vt**2 - a**2) * (Vr + Vt / math.tan(t)) # (3)
                ddt = [dVr_dt, dVt_dt, dp_dt]
                return ddt
            
            # [beta, thetai]
            t = np.linspace(beta, thetai, 50)
            
            sol = solve_ivp(conic_ODEs, [beta, thetai], [ut, -unf, pf], t_eval = t, dense_output='true')
            
            Vt = sol.y[1][-1] # cone surface Vt
            Vr = sol.y[0][-1] # cone surface Vr
            p = sol.y[2][-1] # cone surface p        
        
            # set surface state
            gas.SP = sf, p
            pf = p
            Tf = gas.T
            
            af = eqsound(gas)
            uf = sqrt(Vt**2 + Vr**2)
            Mf = uf / af
            
            df = gas.density
            
            template3 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.6f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template3.format(j-1, beta*180/math.pi, Vt, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            h = 1e-6
            f_prime = (f(beta+h) - f(beta))/h
            
            beta_new = beta - err / f_prime
            
            # convergence check
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % (j-1), end = "\n")
                break
            elif j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            beta = beta_new
            
        print('\n')
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf
    return(M2, u2, d2, p2, T2)
# High Temperature Prandtl-Meyer Expansion
# not available
def pmexpand(M1, u1, d1, p1, T1, theta, perfect='no', imax=500, errmax=0.1):
# convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    gas = ct.Solution('airNASA9-extended.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        print(Fore.YELLOW + "PM Expansion Solution", i+1,"/",ndata, ' - for theta =',thetai)
        
        thetai = math.radians(thetai) # work with radians
        
        # Prandtl-Meyer expansion
        # Isentropic and total enthalpy, H is constant -> state = state(s,h) = state(h)
        
        # upstream state
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        s0 = gas.s              # constant entropy
        ai = eqsound(gas)       # initial speed of sound
        hi = gas.h              # initial enthalpy
        
        ui = Mi * ai
        ht = hi + (ui**2)/2     # total enthalpy is constant
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3f} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Turn Angle', 'error', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', 0, '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        ti = 0          # no itinial inclination
        hstep = 1000
        
        min_tf = 0
        max_tf = 0
        err_old = 0
        tf_old = 0
        
        j = 1
        # reduce h incrementally to turn the flow
        while j < imax + 1:
            j = j + 1
            
            # reduce enthalpy
            hf = hi - hstep
            
            # find new state
            with HiddenPrints():
                cantera_HS(gas, hf, s0)
            
            Tf = gas.T
            pf = gas.P
            df = gas.density
            
            # new velocity
            uf = math.sqrt(2*(ht-hf))
            
            # parameters
            af = eqsound(gas)   # speed of sound
            Mf = uf/af          # downstream mach
            
            # governing equation
            ### print(Mf, uf)
            fVf = math.sqrt(Mf**2 - 1)/uf
            fVi = math.sqrt(Mi**2 - 1)/ui
            
            # reduce or increase hstep
            if (fVf-fVi)/fVi * 100 > 20:
                hstep = hstep * 0.9
            elif (fVf-fVi)/fVi * 100 < 1:
                hstep = hstep * 1.01
            
            dt = (fVf + fVi)/2  * (uf - ui)
            
            tf = ti + dt
            
            err = (tf - thetai)*180/math.pi # in degrees
            
            current_sign = 1 if err > 0 else -1
            
            template3 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.6f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template3.format(j, tf*180/math.pi, err, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            ##### CONVERGENCE AND ITERATION HANDLING
            
            # convergence check
            if abs(err) <= errmax:
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % (j-1), end = "\n")
                break
            elif abs(err) > errmax and j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            # find a range with opposite signs for solution
            # sign check
            
            if 'previous_sign' in locals():
                if current_sign != previous_sign:
                    # opposite signs found, update beta range
                    if current_sign == 1:
                        max_h = hf
                        min_h = hi
                        hstep = (max_h-min_h)/2
                    else:
                        min_h = hf
                        max_h = hi
                        hstep = (max_h-min_h)/2
                else:
                    # no opposite signs, use regular step
                    if current_sign == 1:
                        max_h = hf
                
            # set previos values
            hi = hf
            ui = uf
            Mi = Mf
            ti = tf
            err_old = err
        
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf   
        print('\n')
    return(M2, u2, d2, p2, T2)
    
def pmexpand2(M1, u1, d1, p1, T1, theta, perfect='no', imax=500, errmax=0.1):
# convert scalars into at least 1d array
    M1, u1, d1, p1, T1, theta = np.atleast_1d(M1, u1, d1, p1, T1, theta)
    
    gas = ct.Solution('airNASA9-extended.yaml')
    # gas.TPX = 300.0, ct.one_atm, 'O2:0.21, N2: 0.79'
    # gas.equilibrate('TP')
    
    # Rows will be not be appended iteratively since it is unefficient with numpy
    # Allocate output arrays
    ndata = M1.shape[0]
    M2 = np.zeros(shape=(ndata,)) 
    u2 = np.zeros(shape=(ndata,)) 
    d2 = np.zeros(shape=(ndata,)) 
    p2 = np.zeros(shape=(ndata,)) 
    T2 = np.zeros(shape=(ndata,)) 
    
    # Loop for each data
    # This is necessary since gas properties are taken from cantera
    for i in range(0, ndata):
        # work with scalars
        Mi, ui, di, pi, Ti, thetai = M1[i], u1[i], d1[i], p1[i], T1[i], theta[i]
        
        print(Fore.YELLOW + "PM Expansion Solution", i+1,"/",ndata, ' - for theta =',thetai)
        
        thetai = math.radians(thetai) # work with radians
        
        # Prandtl-Meyer expansion
        # Isentropic and total enthalpy, H is constant -> state = state(s,h) = state(h)
        
        # upstream state
        gas.TP = Ti, pi
        gas.equilibrate('TP')
        s0 = gas.s              # constant entropy
        ai = eqsound(gas)       # initial speed of sound
        hi = gas.h              # initial enthalpy
        
        ui = Mi * ai
        ht = hi + (ui**2)/2     # total enthalpy is constant
        
        # print templates
        template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15} {8:>15} {9:>15} {10:>15}'
        template2 = '{0:>10} {1:>15.3f} {2:>15.3} {3:>15.3f} {4:>15.3} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
        
        print(template.format('Iteration', 'Turn Angle', 'error', 'M2', 'a2', 'u2', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
        print(template2.format('INITIAL', 0, '-', Mi, '-', ui, di, pi, Ti, gas.h, gas.s))
        
        ti = 0          # no itinial inclination
        hstep = 1000
        
        itdir = 0
        err_old = 0
        # reduce h incrementally to turn the flow
        for j in range(1, imax+1):
            # reduce enthalpy
            hf = hi - hstep
            
            # find new state
            with HiddenPrints():
                cantera_HS(gas, hf, s0)
            
            Tf = gas.T
            pf = gas.P
            df = gas.density
            
            # new velocity
            uf = math.sqrt(2*(ht-hf))
            
            # parameters
            af = eqsound(gas)   # speed of sound
            Mf = uf/af          # downstream mach
            
            # governing equation
            print(Mf, uf)
            fVf = math.sqrt(Mf**2 - 1)/uf
            fVi = math.sqrt(Mi**2 - 1)/ui
            
            # reduce or increase hstep
            if (fVf-fVi)/fVi * 100 > 20:
                hstep = hstep * 0.9
            elif (fVf-fVi)/fVi * 100 < 1:
                hstep = hstep * 1.01
            
            dt = (fVf + fVi)/2  * (uf - ui)
            
            tf = ti + dt
            
            err = (tf - thetai)*180/math.pi # in degrees
            
            template3 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.6f} {5:>15.3f} {6:>15.6f} {7:>15.3f} {8:>15.3f} {9:>15.3f} {10:>15.3f}'
            print(template3.format(j, tf*180/math.pi, err, Mf, af, uf, gas.density, gas.P, gas.T, gas.h, gas.s))
            
            if abs(err) <= errmax:
                print(err, errmax)
                print (' -->', Fore.GREEN + 'Converged in %i iterations' % j, end = "\n")
                break
            elif abs(err) > errmax and j == imax:
                print(' -->', Fore.RED + 'Could not converge. Stopped.')
                exit()
            
            # change direction
            if j != 1:
                if abs(err) > abs(err_old):
                    if itdir == 1:
                        hstep = (abs(hstep) - abs(hstep) * err / 100 * 0.001)
                        itdir = 0
                    else:
                        hstep = -(abs(hstep) - abs(hstep) * err / 100 * 0.001)
                        itdir = 1
                        
            
            # set previos values
            hi = hf
            ui = uf
            Mi = Mf
            ti = tf
            err_old = err
        
        M2[i], u2[i], d2[i], p2[i], T2[i] = Mf, uf, df, pf, Tf   
        print('\n')
    return(M2, u2, d2, p2, T2)

# equilibrates mixture with constant entropy and enthalpy
# ONLY USED FOR PM EXPANSION
def cantera_HS(gas, h, s, imax=250, errmax=0.01):
    # downstream conditions
    pi = gas.P
    hf = h
    si = s

    """
    Iteration parameters
    - Initial guess of pressure ratio is made from isentropic perfect gas relation for k = 1.4.
    - imax is max number of iterations. Given in function definition.
    - errmax is max relative error percent. Given in function definition.
    """       
    
    pratio = 0.1
    itdir = 1 # iteration direction
    err_old = 0
    
    template = '{0:>10} {1:>15} {2:>15} {3:>15} {4:>15} {5:>15} {6:>15} {7:>15}'
    print(template.format('Iteration', 'pratio', 'rel_rror', 'd2', 'p2', 'T2', 'h2', 's2'), '\n', '-'*169)
    
    # Iterating to converge
    for j in range(1, imax+1):
        pf = pi/pratio
        gas.HP = hf, pf
        
        
        gas.equilibrate('HP')

        sf = gas.s
        df = gas.density

        err = (sf - si)/si * 100
        
        df = gas.density
        Tf = gas.T
       
        template2 = '{0:>10} {1:>15.3f} {2:>15.3f} {3:>15.3f} {4:>15.6f} {5:>15.6f} {6:>15.6f} {7:>15.3f}'
        print(template2.format(j, pratio, err, df, pf, Tf, hf, sf))
        
        # convergence criterion
        if abs(err) <= errmax:
            print ('-->', 'Converged in ', j, ' iterations', end = "\n\n") 
            break
        elif abs(err) > errmax and j == imax:
            print('-->', 'Could not converge. Stopped.')
            exit()
            
        # iteration direction      
        if j != 1:
            if abs(err) > abs(err_old):
                if itdir == 1:
                    itdir = 0
                else:
                    itdir = 1

        # iteration steps
        if itdir == 1:
            pratio = pratio - 2 * pratio * err / 100
        else:
            pratio = pratio + 2 * pratio * err / 100

        # old error
        err_old = err
    print('-' * 50, end = '\n')
    return gas

# One dimensional steady gas network creator
# initial inputs are arrays
# not available atm
def gas1d(initial, gas_network):
    inputs = initial
    for func, angle in gas_network:
        if func.__name__ in ['oshock', 'cshock', 'pmexpansion']:
            inputs = func(inputs, angle)
        else:
            inputs = func(inputs)
    return inputs

# to enable/disable prints

# disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# enable printing
def enablePrint():
    sys.stdout = sys.__stdout__

# Execute examples
if __name__ == "__main__":

    
    u1 = 10972.8/29.05
    M1 = u1 / sqrt(1.4 * 287.1 * 268.41)
    """
    # nshock(np.array([M1,M1*2]), np.array([10972.8, 2*10972.8]), np.array([0.00078192, 0.00078192]), np.array([60.484, 60.484]), np.array([268.41, 268.41]))
    # print('-' * 50, end = '\n\n')
    
    # Single normal shock example
    time1 = time.process_time()
    nshock(M1, u1, 0.00078192, 60.484, 268.41)
    time2 = time.process_time()
    print(time2-time1)
    
    time1 = time.process_time()
    nshock_SCPY(M1, u1, 0.00078192, 60.484, 268.41)
    nshock_SCPY(M1, u1, 0.00078192, 60.484, 268.41)
    time2 = time.process_time()
    print(time2-time1)
    """
    
    # stagnation(0.572, 242.84, 3.401, 441154.52, 450.02)
    # stagnation_NRS(0.572, 242.84, 3.401, 441154.52, 450.02)
    
    """
    print('-' * 50, end = '\n\n')
    stagnation(*nshock(M1, u1, 0.00078192, 60.484, 268.41))
    print('-' * 50, end = '\n\n')
    stagnation(M1, u1, 0.00078192, 60.484, 268.41)
    """
    
    """
    nshock(M1, 10972.8, 0.00078192, 60.484, 268.41)
    print('-' * 50, end = '\n\n')
    stagnation(0.2215, 717.547, 0.012, 88049, 11633.63)
    print('-' * 50, end = '\n\n')
    stagnation(M1, 10972.8, 0.00078192, 60.484, 268.41)

    

    # stagnation(M1, 10972.8, 0.00078192, 60.484, 268.41)
    # oshock(6, 1989, 0.021630, 1696.4, 273.23, 30)
    # oshock(np.array([M1, M1*1.9]), np.array([10972.8, 1.9*10972.8]), np.array([0.00078192, 0.00078192]), np.array([60.484, 60.484]), np.array([268.41, 268.41]), np.array([30, 30]))
    
    
    M1 = 2
    M2 = 0.3
    T1 = 267.5
    T2 = 267.5
    u1 = M1 * sqrt(1.4 *  287.1 * T1)
    u2 = M2 * sqrt(1.4 *  287.1 * T2)
    
    nshock(M1, u1, 1.2596, 100000, T1)
    stagnation(0.572, 242.84, 3.401, 441154.52, 450.02)
    print('-' * 50, end = '\n\n')
    stagnation(M1, u1, 1.2596, 100000, T1)
    print('-' * 50, end = '\n\n')
    
    # stagnation(np.array([M1, M2]), np.array([u1, u2]), np.array([1.2596, 1.2596]), np.array([100000, 100000]), np.array([T1, 1500]))
    
    # nshock(M1, 10972.8, 0.00078192, 60.484, 268.41)
    
    # print('NEW', end = '\n')
    # nshock(4000, 0.186481, 11597.3, 216.65)
    """
    nshock(28, 28*sqrt(1.4*287.1*283), 6.33e-4, 48.13, 283)
    # nshock_SCPY(16, 16*sqrt(1.4*287.1*283), 6.33e-4, 48.13, 283)
    # _,_,T0r = stagnation(*nshock(32.5, 32.5*sqrt(1.4*287.1*283), 6.33e-4, 48.13, 283))
    # pnshockstag(32.5, 32.5*sqrt(1.4*287.1*283), 6.33e-4, 48.13, 283)
    
    
    # Normal and stagnation for a range of data example
    """
    # input for comparison
    num = 101
    Min = np.linspace(1, 12, num)
    Tin = np.repeat(300, num)
    uin = Min * np.sqrt(1.4 * 287.1 * Tin)
    pin = np.repeat(101325, num)
    din = np.repeat(1.225, num)
    nshock(Min, uin, din, pin, Tin)
    """
    """
    stagnation(*nshock(Min, uin, din, pin, Tin))
    
    """
    
    """
    print(Min)
    # print(Min, uin, din, pin, Tin)
    print('-' * 50, end = '\n\n')
    # nshock(10, 10*sqrt(1.4 * 287.1 * 300), 1.225, 101325, 300)
    # nshock(1.1, 1.1*sqrt(1.4 * 287.1 * 300), 1.17661, 101325, 300)
    
    # nshock(Min, uin, din, pin, Tin)
    # _,_,T0 = stagnation(*nshock(Min, uin, din, pin, Tin))
    
    print('-' * 50, end = '\n\n')
    # nshock(1.1, 1.1*sqrt(1.4 * 287.1 * 300), 1.17661, 101325, 300, perfect='yes')
    # nshock(np.array([1.1, 1.5]), np.array([1.1*sqrt(1.4 * 287.1 * 300), 1.5*sqrt(1.4 * 287.1 * 300)]), np.array([1.17661,1.17661]), np.array([101325,101325]), np.array([300,300]), perfect='yes')
    
    M2, T0p = pnshockstag(Min, uin, din, pin, Tin)
    print(M2)
    print(T0p)
    
    _,_,T0r = stagnation(*nshock(Min, uin, din, pin, Tin))
    
    _,_,T0s = stagnation(Min, uin, din, pin, Tin)
    
    fig, ax = plt.subplots()
    ax.plot(M2, T0p, label='Perfect Gas Air')
    ax.plot(Min, T0r, label='High Temperature Gas Air-1 (After Shock)')
    ax.plot(Min, T0s, label='High Temperature Gas Air-2')
    ax.set_xlabel('Shock Uptream Mach Number')
    ax.set_ylabel('Stagnation Temperature (K)')
    ax.set_title('Comparison of Perfect and High Temperature Air Models')
    ax.xaxis.set_ticks(np.arange(0, 31, 2))
    # ax.yaxis.set_ticks(np.arange(0, 60000, 5000))
    ax.legend()
    ax.grid(visible=True_)
    plt.show()
    """
    
    # oshock(3, 3*sqrt(1.4 * 287.1 * 288.15), 1.225, 101325, 288.15, 15)
    # oshock(6, 1989, 0.021630, 1696.4, 273.23, 30)
    # oshock_NRS(6, 1989, 0.021630, 1696.4, 273.23, 30)
    # Oblique shock single example
    
    
    """
    exit
    thetain = np.repeat(20, num)
    oshock(Min, uin, din, pin, Tin, thetain)
    exit
    cst = 1
    oshock(6*cst, 1989*cst, 0.021630, 1696.4, 273.23, 43)
    oshock(np.array([M1, M1*1.9]), np.array([10972.8, 1.9*10972.8]), np.array([0.00078192, 0.00078192]), np.array([60.484, 60.484]), np.array([268.41, 268.41]), np.array([30, 30]))
    """
    ###############################################
    
    # Oblique shock examples
    
    #oshock(2, 2*343, 1.225, 101325, 288.15, 20)
    #oshock(10, 10*343, 1.225, 101325, 288.15, 15)
    
    # 100000 feet altitude - 10000 feet/sec - Anderson page 614
    #oshock(3*1524/sqrt(1.4 * 287.1 * 227.1), 3*1524, 0.00759, 1090, 227.1, 8)
    #oshock(3*1524/sqrt(1.4 * 287.1 * 227.1), 3*1524, 0.00759, 1090, 227.1, 8)
    
    ###############################################
    
    # Conic Shock examples
    
    # cshock(10, 10*sqrt(1.4 * 287.1 * 288.15), 1.225, 101325, 288.15, 15)
    # cshock_NRS(10, 10*sqrt(1.4 * 287.1 * 288.15), 1.225, 101325, 288.15, 15)
    # nshock(20, 20*340, 1.225, 101325, 288.15)
    # cshock(12, 12*340, 1.225, 101325, 300, 15)
    
    # cshock(2, 2*sqrt(1.4 * 287.1 * 288.15), 1.225, 101325, 288.15, 30)
    
    # Nasa report verified.
    
    # cshock(3.506307000000000063e+00, 1.057300411258018130e+03, 1.905350090368200175e-02, 1.237498113810740278e+03, 2.263012706133699794e+02, 15.30)
    
    # cshock(2.268, 771.718, 1.225, 101325, 288.15, 15.30)
    # cshock(1.034, 344.585, 1.0268, 81502.818, 276.408, 15.3)
    
    ###############################################
    
    # Prandtl - Meyer examples
    
    # pmexpand(1, 806.52, 3.8073, 1907300, 1738.3, 45)
    # pmexpand(2, 2*math.sqrt(1.4*287*290), 1.225, 101325, 290, 10)
    # pmexpand(1, 806.52, 3.8231, 1907300, 1738.3, 45)
    # pmexpand(1, 1*math.sqrt(1.4*287*6140), 0.088373, 1.2*101325, 6140, 60)
    # pmexpand(6, 6*math.sqrt(1.4*287*290), 1.225, 101325, 6140, 50)
    
    # pmexpand(1.000, 1528.489,0.088373,121590.000,6140.000,21.65)
    
    ###############################################
    # EXTRAS
    
    # cshock(8.5, 8.5*sqrt(1.4*287*300), 1.225, 101325, 300, 15.30)
    # cshock(2, 2*sqrt(1.4*287*300), 1.225, 101325, 300, 15.30)
    
    # 381           0.049           0.583      414.167051         241.636        2.029564      249954.103         429.125      132207.902        6970.653
    # nshock(6, 6*317, 0.71532, 51325, 250, perfect='yes')
    
    # cshock(1.158, 384.350, 0.989473, 77853.668, 274.014, 7.33)
    # stagnation(*nshock(1.158, 1.158*331.9, 0.989473, 77853.668, 274.014))
    
    # oshock(2, 347*1.5, 1.225, 101325, 300, 7.3, perfect='no', imax=200, errmax=0.1)
    
    """
    nshock(2.16, 742.079, 0.9783515, 68452.7, 279.34, perfect='yes')
    stagnation(*nshock(5.25, 1374.81, 0.040285, 1972.59, 170.61))
    cshock(5.25, 1374.81, 0.04028530484047352, 1972.59, 170.61, 15)
    oshock(5.25, 1374.81, 0.04028530484047352, 1972.59, 170.61, 60)
    
    
    Rnose = 0.0254
    pfree = 1374.81
    result = htl.propertyRatioHT(3.982, 267, 7803, 0.1)
    print(result)
    """
    
    # cshock(5.25, 1374.81, 0.04028530484047352, 1972.59, 170.61, 15)
    # pmexpand(1.5, 1.5*347, 1.225, 101325, 300, 5)
    # M = 1.5
    # a = cshock(M, M*math.sqrt(287*1.4*275.94), 1.019268, 80755.176, 275.940, 15.3)
    # print(a)
    
    """
    cshock(2.441,771.751,0.655310, 46798.364,248.742, 15.3)
    cshock(1.737, 564.679, 0.832534, 62901.745, 263.119, 15.3)
    """
    
    
    """
    num = 21
    Min = np.repeat(1, num)
    Tin = np.repeat(6140, num)
    uin = Min   # doesnt matter
    pin = np.repeat(1.2 * 101325, num)
    din = np.repeat(0.088373, num)
    thin = np.linspace(1, 60, num)
    
    
    _, _, _, p, _ = pmexpand(Min, uin, din, pin, Tin, thin)
    
    ratio = np.divide(p, pin)
    print(ratio)
    
    """