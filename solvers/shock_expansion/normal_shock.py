import cantera as ct
import numpy as np
from tqdm import tqdm
from scipy.optimize import newton

# import within module library
from utilities import equilibrium_speed_of_sound as eqsound 

def solve_normal_shock(M1, T, p, method='equilibrium', gas='airNASA9.yaml', verbose=1, rtol_shock=1e-6, max_iter=200):
    M1, T, p = np.atleast_1d(M1), np.atleast_1d(T), np.atleast_1d(p) # Ensure inputs are at least 1D arrays
    ndata = M1.size                                                  # Number of points to process
    
    # Error handling for input validation
    if ndata == 0:
        raise ValueError("Input arrays M1, T, and p must not be empty.")
    if M1.size != T.size or M1.size != p.size:
        raise ValueError("Input arrays M1, T, and p must have the same shape.")
    if M1.ndim > 1 or T.ndim > 1 or p.ndim > 1:
        raise ValueError("Input arrays M1, T, and p must be 1D arrays.")
    if not isinstance(rtol_shock, (float, int)) or rtol_shock <= 0:
        raise ValueError("Relative tolerance 'rtol_shock' must be a positive number.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations 'max_iter' must be a positive integer.")
    if method not in ['auto', 'equilibrium']:
        raise ValueError("Method must be either 'auto' or 'equilibrium'.")
    if gas is None:
        raise ValueError("Must provide a gas object or a mechanism file.")
    if verbose not in [0, 1, 2]:
        raise ValueError("Verbose must be 0 (silent), 1 (progress bar), or 2 (detailed).")
    if isinstance(gas, str):
        # If gas is a string, assume it's a Cantera mechanism file
        try:
            gas = ct.Solution(gas)
        except Exception as e:
            raise ValueError(f"Failed to load gas model '{gas}': {e}")
    elif not isinstance(gas, ct.Solution):
        raise TypeError("Gas must be a Cantera Solution object or a string representing a mechanism file.")
   
    # Pre-allocate output arrays
    M2 = np.zeros_like(M1, dtype=float)
    T2 = np.zeros_like(M1, dtype=float)
    p2 = np.zeros_like(M1, dtype=float)

    iterator = range(ndata) # If verbose >= 1, use tqdm for progress bar
    if verbose >= 1 and ndata > 1:
        iterator = tqdm(range(ndata), desc="Solving Normal Shock")
    
    for i in iterator:
        try:
            Mi, Ti, pi = M1.item(i), T.item(i), p.item(i)

            # Set the gas state
            gas.TP = Ti, pi
            gas.equilibrate('TP')

            di = gas.density        # Density at upstream state
            hi = gas.h              # Enthalpy at upstream state
            ai = eqsound(gas)       # Speed of sound at upstream state
            ui = Mi * ai            # Upstream velocity

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
            
            dratio = dratio_new # Update dratio with the newton method result
            pf = pi + di * (ui**2) * (1-dratio)
            hf = hi + (ui**2)/2 * (1-dratio**2)
            gas.HP = hf, pf
            gas.equilibrate('HP')
            df = gas.density
            Tf = gas.T

            uf = di * ui / df
            af = eqsound(gas)
            Mf = uf/af

            # Store results
            M2[i], T2[i], p2[i] = Mf, Tf, pf           

        except RuntimeError as e: # Catch the specific error from the solver
            # Log a Warning to the console
            if verbose >=1:
                tqdm.write(f"Warning: Convergence failed at point {i} (M1={Mi:.2f}). Skipping.")
            
            # Flag the output data with NaN
            M2[i], T2[i], p2[i] = np.nan, np.nan, np.nan

    return M2, T2, p2

def solve_normal_shock_perfect(M1, T, p, k=1.4, verbose=1):
