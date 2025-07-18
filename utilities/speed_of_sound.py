import math

def equilibrium_speed_of_sound(gas=None, T=None, p=None, rtol=1e-12, max_iter=5000):
    """
    Calculates the equilibrium speed of sound without changing the gas state.
    If gas is not provided, T (temperature in K) and p (pressure in Pa) must be provided,
    and a new Cantera gas object will be created.

    Parameters
    ----------
    gas : Cantera.Solution, optional
        The gas object whose equilibrium speed of sound is to be calculated.
    T : float, optional
        Temperature in K (required if gas is not provided).
    p : float, optional
        Pressure in Pa (required if gas is not provided).
    rtol : float, optional
        Relative tolerance for the equilibrium calculation (default: 1e-12).
    max_iter : int, optional
        Maximum number of iterations for the equilibrium calculation (default: 5000).

    Returns
    -------
    soundequil : float
        The equilibrium speed of sound in m/s.
    """
    import cantera as ct

    # If gas is not provided, create a new Cantera gas object
    if gas is None:
        if T is None or p is None:
            raise ValueError("If 'gas' is not provided, both 'T' and 'p' must be specified.")
        gas = ct.Solution('airNASA9-transport.yaml')  # Need a global mechanism description
        gas.TP = T, p
        gas.equilibrate('TP')

    # Store the original thermodynamic state
    h0 = gas.h
    p0 = gas.P

    try:
        # Extract entropy and density at the original state for isentropic perturbation
        s0 = gas.s
        rho0 = gas.density

        # Perturb pressure
        p1 = p0 * 1.00001

        # Set new state at constant entropy and re-equilibrate
        gas.SP = s0, p1
        gas.equilibrate('SP')

        # Calculate speed of sound from the finite difference
        denominator = gas.density - rho0
        if denominator <= 0:
            raise ValueError("Density difference is zero or negative; cannot compute speed of sound.")
        soundequil = math.sqrt((p1 - p0) / denominator)
    except Exception as e:
        raise RuntimeError("Error calculating speed of sound.") from e
    finally:
        # Restore the gas object to its original state
        gas.HP = h0, p0
        gas.equilibrate('HP')

    return soundequil