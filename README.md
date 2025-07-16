# htshock
# HTShock: Real Gas Equilibrium Shock Solver

A robust Python-based tool for calculating the post-shock conditions of equilibrium air across a normal shock wave, accounting for real gas effects. `HTShock` leverages the high-fidelity thermodynamic properties from the **Cantera** library and employs a **Newton-Raphson** method to solve the Rankine-Hugoniot equations.

This solver is designed for applications in hypersonic aerodynamics, re-entry vehicle analysis, and high-temperature gas dynamics where the ideal gas assumption is no longer valid.

## Key Features

* **Real Gas Physics:** Accurately models equilibrium air by considering the dissociation of N₂ and O₂ and the formation of nitric oxide (NO).

* **High-Fidelity Thermodynamics:** Uses the **Cantera** library to handle complex thermodynamic state calculations, ensuring accurate properties for high-temperature air mixtures.

* **Robust Numerical Solver:** Implements a Newton-Raphson iterative method to solve the non-linear system of Rankine-Hugoniot equations.

* **Comprehensive Output:** Calculates all key post-shock properties, including pressure, temperature, density, velocity, and equilibrium species mole fractions.

## Governing Equations

The solver finds the post-shock state (2) from a given pre-shock state (1) by solving the Rankine-Hugoniot relations for the conservation of mass, momentum, and energy:

* **Mass Conservation:**
  
$$
\rho_1 u_1 = \rho_2 u_2
$$

* **Momentum Conservation:**
  
$$
P_1 + \rho_1 u_1^2 = P_2 + \rho_2 u_2^2
$$

* **Energy Conservation:**
  
$$
h_1 + \frac{u_1^2}{2} = h_2 + \frac{u_2^2}{2}
$$

Here, $\rho$ is the density, $u$ is the velocity, $P$ is the pressure, and $h$ is the specific enthalpy. The specific enthalpy $h$ is a complex function of temperature and composition, which is where Cantera's powerful thermodynamic engine is essential.

## Installation

To get started with `HTShock`, you need Python and Cantera installed. A virtual environment is highly recommended.

**1. Prerequisites:**

* Python (3.8 or newer)

* Cantera: Follow the official installation guide at [cantera.org/install](https://cantera.org/install/index.html). Using Conda is often the easiest method: