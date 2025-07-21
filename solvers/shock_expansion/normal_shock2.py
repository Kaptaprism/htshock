""""""
def solve_normal_shock2(M1, gas=None, T=None, p=None, rtol_shock=1e-6, max_iter=50):
    """
    Calculates downstream conditions for a normal shock in equilibrium air.

    Accepts either a pre-made Cantera gas object or NumPy arrays for M1, T, and p.
    """
    
    # --- 1. Input Handling and Validation ---
    if T is not None and p is not None:
        # Convert all inputs to at least 1D NumPy arrays
        M1 = np.atleast_1d(M1)
        T = np.atleast_1d(T)
        p = np.atleast_1d(p)
        
        if not (M1.shape == T.shape == p.shape):
            raise ValueError("Input arrays M1, T, and p must have the same shape.")
            
        # Create the gas object ONCE for efficiency
        gas = ct.Solution('airNASA9.yaml')
        is_array_mode = True
        num_points = M1.size
        
    elif gas is not None:
        M1 = np.atleast_1d(M1)
        if M1.size > 1:
            raise ValueError("Cannot process an array of Mach numbers with a single gas state.")
        is_array_mode = False
        num_points = 1
    else:
        raise ValueError("Must provide either a 'gas' object or both 'T' and 'p'.")

    # --- 2. Pre-allocate Output Arrays ---
    # This is much more efficient than appending to lists in a loop
    M2_out = np.zeros_like(M1, dtype=float)
    T2_out = np.zeros_like(M1, dtype=float)
    p2_out = np.zeros_like(M1, dtype=float)
    # Add other outputs as needed (u2, d2, etc.)

    # --- 3. The Main Calculation Loop ---
    for i in range(num_points):
        # Extract scalar values for the current point
        Mi = M1.item(i) # .item() extracts the scalar value
        
        if is_array_mode:
            # If in array mode, set the upstream state for each point
            Ti = T.item(i)
            pi = p.item(i)
            gas.TP = Ti, pi
            gas.equilibrate('TP')

        # Get upstream properties for the current point
        h1 = gas.h
        d1 = gas.density
        a1 = eqsound(gas)
        u1 = Mi * a1
        
        # --- Root finding logic for this single point ---
        # (This is the same logic from our previous discussions)
        def _residual(d_ratio_inv, u1, d1, p1, h1, gas_obj):
            d2_assumed = d1 * d_ratio_inv
            p2_calc = p1 + d1 * u1**2 * (1 - 1 / d_ratio_inv)
            h2_calc = h1 + 0.5 * u1**2 * (1 - (1 / d_ratio_inv)**2)
            gas_obj.HP = h2_calc, p2_calc
            gas_obj.equilibrate('HP')
            return (gas_obj.density - d2_assumed) / d2_assumed

        gamma_guess = 1.4 # Use perfect gas for initial guess
        d_ratio_inv_guess = ((gamma_guess + 1) * Mi**2) / (2 + (gamma_guess - 1) * Mi**2)
        
        try:
            converged_d_ratio_inv = newton(_residual, x0=d_ratio_inv_guess, 
                                           args=(u1, d1, gas.P, h1, gas), 
                                           tol=rtol_shock, maxiter=max_iter)
            
            p2 = gas.P
            T2 = gas.T
            a2 = eqsound(gas)
            u2 = u1 / converged_d_ratio_inv
            M2 = u2 / a2

            # Store results in the output arrays
            M2_out[i] = M2
            T2_out[i] = T2
            p2_out[i] = p2

        except RuntimeError:
            # Handle non-convergence for a point, e.g., store NaN
            M2_out[i], T2_out[i], p2_out[i] = np.nan, np.nan, np.nan
            print(f"Warning: Convergence failed for point {i} (M1={Mi:.2f})")

    # --- 4. Return the Results ---
    # Return scalars if input was scalar, otherwise return arrays
    if M1.size == 1 and T is None:
        return M2_out.item(), T2_out.item(), p2_out.item()
    else:
        return M2_out, T2_out, p2_out
""""""