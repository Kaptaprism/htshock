import cantera as ct
import numpy as np
import math
from utilities import equilibrium_speed_of_sound as eqsound
from solvers.shock_expansion import solve_normal_shock as nshock

# print(eqsound.__doc__)    



myair = ct.Solution('airNASA9-transport.yaml')
myair.TP = 300, 101325
myair.equilibrate('TP')

"""
myair.TPX = 300, 101325, 'O2:0.21, N2:0.79'

a = eqsound(myair) # type: ignore
b = math.sqrt(1.4* 287.05 * myair.T) # Speed of sound in air at 300 K

print(f"Calculated speed of sound: {a:.2f} m/s")
print(f"Expected speed of sound: {b:.2f} m/s")
# Expected output: 340.29 m/s (approximately)
"""

# Define a large range of Mach numbers, temperatures, and pressures for sampling
mach_numbers = np.linspace(1.5, 20, 50)  # 100 samples from Mach 1.5 to 20
temperatures = np.full_like(mach_numbers, 300.0)  # All at 300 K
pressures = np.full_like(mach_numbers, 101325.0)  # All at 101325 Pa

# nshock(M1=mach_numbers, T=temperatures, p=pressures, verbose=2, gas=myair)
M2 = nshock(1, 300, 101325, verbose=2)
print(M2)

"""
for i in range(len(mach_numbers)):
    print(f"M1={mach_numbers[i]:.1f} -> T2={T2_arr[i]:.1f} K, p2={p2_arr[i]/1e5:.2f} bar")
"""