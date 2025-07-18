# shock_solver_module.py
# Contains the core classes for flow state representation and shock calculations.
# VERSION 3: Handles CanteraErrors for a fully robust nshock solver.

import cantera as ct
import numpy as np
import math
from scipy.optimize import root_scalar

class FlowState:
    """A class to hold the thermodynamic and kinematic state of a flow."""
    def __init__(self, M, u, P, T, rho, gas_model_name):
        self.M = M
        self.u = u
        self.P = P
        self.T = T
        self.rho = rho
        self.gas_model_name = gas_model_name

    def __repr__(self):
        """Provides a clean string representation of the object."""
        return (f"FlowState(M={self.M:.3f}, u={self.u:.1f} m/s, "
                f"P={self.P:.1f} Pa, T={self.T:.1f} K, rho={self.rho:.6f} kg/m^3)")

    def to_dict(self):
        """Converts the object's properties to a dictionary for data export."""
        return {
            'Mach': self.M,
            'Velocity (m/s)': self.u,
            'Pressure (Pa)': self.P,
            'Temperature (K)': self.T,
            'Density (kg/m^3)': self.rho
        }

class ShockSolver:
    """A solver class for high-temperature shock calculations."""
    def __init__(self, gas_yaml_path):
        """Initializes the solver with a Cantera gas object."""
        self.gas = ct.Solution(gas_yaml_path)
        self.gas_model_name = gas_yaml_path

    def _get_equilibrium_sound_speed(self):
        """
        Calculates equilibrium speed of sound by perturbing pressure at constant entropy.
        """
        # This function is assumed to be called when self.gas is already in a valid state.
        s0 = self.gas.s
        p0 = self.gas.P
        rho0 = self.gas.density
        p1 = p0 * 1.0001
        self.gas.SP = s0, p1
        self.gas.equilibrate('SP')
        return math.sqrt((p1 - p0) / (self.gas.density - rho0))

    def nshock(self, upstream: FlowState, solver_params: dict):
        """
        Calculates the downstream state across a normal shock using a robust, bracketed solver.
        """
        if upstream.M < 1.05:
            return upstream

        self.gas.TP = upstream.T, upstream.P
        h_up = self.gas.h

        # --- IMPROVEMENT: This function is now fully robust ---
        def error_function(dratio, u_up, rho_up, h_up, p_up):
            p_down = p_up + rho_up * (u_up**2) * (1 - dratio)
            h_down = h_up + (u_up**2) / 2 * (1 - dratio**2)
            
            if p_down <= 0: return 1.0e9

            try:
                # This is the line that was causing the CanteraError
                self.gas.HP = h_down, p_down
                # No need to call equilibrate if HP is set, as HP sets the equilibrium state
            except ct.CanteraError:
                # If Cantera fails for this H,P pair, it's an invalid dratio.
                # Return a large error value to force the solver to try a different dratio.
                return 1.0e9
            
            return self.gas.density - rho_up / dratio

        bracket = [0.01, 0.99]
        
        try:
            sol = root_scalar(
                error_function,
                args=(upstream.u, upstream.rho, h_up, upstream.P),
                method='brentq',
                bracket=bracket,
                xtol=solver_params['error_tolerance']
            )
            if not sol.converged:
                raise RuntimeError(f"Solver failed to converge: {sol.flag}")
            dratio_sol = sol.root

        except (RuntimeError, ValueError) as e:
            # This will catch failures of the root_scalar itself (e.g., if a bracket is invalid)
            print(f"Solver failed: {e}")
            return None

        # --- Post-processing and Output ---
        rho_down = upstream.rho / dratio_sol
        u_down = upstream.u * dratio_sol
        p_down = upstream.P + upstream.rho * (upstream.u**2) * (1 - dratio_sol)
        h_down = h_up + (upstream.u**2) / 2 * (1 - dratio_sol**2)

        self.gas.HP = h_down, p_down
        T_down = self.gas.T
        
        a_down = self._get_equilibrium_sound_speed()
        M_down = u_down / a_down

        return FlowState(M=M_down, u=u_down, P=p_down, T=T_down, rho=rho_down, gas_model_name=self.gas_model_name)