import cantera as ct
import numpy as np
import math
from utilities import equilibrium_speed_of_sound as eqsound
from solvers.shock_expansion import solve_normal_shock as nshock

# print(eqsound.__doc__)    

myair = ct.Solution('airNASA9-transport.yaml')
myair.TP = 300, 101325
myair.equilibrate('TP')

myair.TPX = 300, 101325, 'O2:0.21, N2:0.79'

a = eqsound(myair) # type: ignore
b = math.sqrt(1.4* 287.05 * myair.T) # Speed of sound in air at 300 K

print(f"Calculated speed of sound: {a:.2f} m/s")
print(f"Expected speed of sound: {b:.2f} m/s")
# Expected output: 340.29 m/s (approximately)

mach_numbers = np.array([5, 10, 15])
temperatures = np.array([300, 400, 500]) # Kelvin
pressures = np.ones(3) * 101325 # Pascals

M2_arr, T2_arr, p2_arr = nshock(M1=mach_numbers, T=temperatures, p=pressures)
for i in range(len(mach_numbers)):
    print(f"M1={mach_numbers[i]:.1f} -> T2={T2_arr[i]:.1f} K, p2={p2_arr[i]/1e5:.2f} bar")