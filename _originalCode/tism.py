# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:30:15 2025

@author: DOGAN.AKCAKAYA
"""

#%% import modules and load std atmosphere
# import modules
import htshock2
import htcorrelate
import htransport

import sys, os
import cantera as ct
import numpy as np
import csv
import matplotlib.pyplot as plt

from numpy import *
from scipy.interpolate import interp1d

###############################################################################
# functions can be executed in this class to supress their outputs
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# load std atmosphere
with open('std_atm.txt', 'r') as f:
    std_atm = list(csv.reader(f, delimiter='\t'))

std_atm = np.array(std_atm, dtype = float)
std_atm[:, 1], std_atm[:, 3] = std_atm[:, 1] * 101325, std_atm[:, 3] * 1.225
std_atm = np.delete(std_atm, 4, 1)
###############################################################################
#%% load flight trajectory data
with open('ISMpTayfun_IVK.csv', 'r') as f:
    data = list(csv.reader(f, delimiter=';'))
del data[0]
flight_data_original = np.array(data, dtype = float)
###############################################################################
#%% plot flight data (optional)
time_original, mach_original, altitude_original = flight_data_original[:,0], flight_data_original[:,1], flight_data_original[:,2]
color = 'tab:blue'
fig, ax1 = plt.subplots()
ax1.plot(time_original, mach_original, label='Mach')
ax1.set_xlabel('Zaman(s)')
ax1.set_ylabel('Mach')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(visible=True_)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.plot(time_original, altitude_original, label='İrtifa', color = color)
ax2.set_ylabel('İrtifa (m)', color = color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(0,900, step=100))
ax1.set_title('Mach ve İrtifa vs Zaman')
ylim1 = ax1.get_ylim()
len1 = ylim1[1]-ylim1[0]
yticks1 = ax1.get_yticks()
rel_dist = [(y-ylim1[0])/len1 for y in yticks1]
ylim2 = ax2.get_ylim()
len2 = ylim2[1]-ylim2[0]
yticks2 = [ry*len2+ylim2[0] for ry in rel_dist]

ax2.set_yticks(yticks2)
ax2.set_ylim(ylim2)
plt.show()
###############################################################################
#%% downsample data
timestep = 0.5
ftime = flight_data_original[:, 0]
fdata = flight_data_original[:, 1:]

ftime_new = np.arange(ftime[0], ftime[-1], timestep)
ftime_new = np.append(ftime_new, ftime[-1])

fdata_new = np.array([
    np.interp(ftime_new, ftime, fdata[:, i]) for i in range(fdata.shape[1])
]).T

flight_data = np.column_stack((ftime_new, fdata_new))
np.savetxt('.\\output\\downsampled_flight.txt', flight_data, delimiter=';')
###############################################################################
#%% interpolate standart atmoshpere
ndata = flight_data.shape[0]
std_altitude = std_atm[:,0]
flight_altitude = flight_data[:,2]

interpolated_columns = []
for i in range(0, std_atm.shape[1]):
    interpolator = interp1d(std_altitude, std_atm[:,i], kind='linear', fill_value = 'extrapoalte')
    interpolated_col = interpolator(flight_altitude)
    interpolated_columns.append(interpolated_col)
flight_expanded = np.column_stack(interpolated_columns)

# calculate velocity and combina data to single, and at the end reorder.
velocity = flight_data[:,1] * np.sqrt(1.4 * flight_expanded[:,2] * 287)
flight_data = np.column_stack([flight_data, velocity])

data = np.concatenate((flight_data[:,[0, 1, 3]], flight_expanded[:, 1:4]), 1)
data = data[:, [0, 1, 2, 5, 3, 4]]

np.savetxt('.\\output\\input_data.txt', data)

#%% solve conic 
from htshock2 import cshock, cshock_NRS

alpha = 15.302                          # cone angle
result1 = np.zeros(shape=(ndata, 5))    # result for after conic

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if data[i][1] > 1.01:
        with HiddenPrints():
            result1[i][0], result1[i][1], result1[i][2], result1[i][3], result1[i][4] = cshock(data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], alpha)
    else:   # subsonic region does not change significantly.
        print('CSHOCK SUBSONIC')
        with HiddenPrints():
            result1[i][0], result1[i][1], result1[i][2], result1[i][3], result1[i][4] = data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]

np.savetxt('.\\output\\cshock.txt', result1)

#%% solve pm expansion after conic
from htshock2 import pmexpand
alpha1 = 9.650
result2 = np.zeros(shape=(ndata, 5))    # result for after pm

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if result1[i][0] > 1.15:
        with HiddenPrints():
            result2[i][0], result2[i][1], result2[i][2], result2[i][3], result2[i][4] = pmexpand(result1[i][0], result1[i][1], result1[i][2], result1[i][3], result1[i][4], alpha1, errmax=0.5)
    else:   # subsonic region does not change significantly.
        print('PM SUBSONIC')
        with HiddenPrints():
            result2[i][0], result2[i][1], result2[i][2], result2[i][3], result2[i][4] = result1[i][0], result1[i][1], result1[i][2], result1[i][3], result1[i][4]

np.savetxt('.\\output\\pm1.txt', result2)
#%% solve pm expansion after pm
from htshock2 import pmexpand
alpha2 = 5.652
result3 = np.zeros(shape=(ndata, 5))    # result for after pm

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if result2[i][0] > 1.15:
        with HiddenPrints():
            result3[i][0], result3[i][1], result3[i][2], result3[i][3], result3[i][4] = pmexpand(result2[i][0], result2[i][1], result2[i][2], result2[i][3], result2[i][4], alpha2, errmax=0.5)
    else:   # subsonic region does not change significantly.
        print('PM SUBSONIC')
        with HiddenPrints():
            result3[i][0], result3[i][1], result3[i][2], result3[i][3], result3[i][4] = result2[i][0], result2[i][1], result2[i][2], result2[i][3], result2[i][4]

np.savetxt('.\\output\\pm2.txt', result3)
#%% solve stagnation
from htshock2 import nshock, nshock_SCPY
 
result4 = np.zeros(shape=(ndata, 5))    # result for stag

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if data[i][1] > 1.15:
        with HiddenPrints():
            result4[i][0], result4[i][1], result4[i][2], result4[i][3], result4[i][4] = nshock(data[i][1], data[i][2], data[i][3], data[i][4], data[i][5])
    else:   # subsonic region does not change significantly.
        print('PM SUBSONIC')
        with HiddenPrints():
            result4[i][0], result4[i][1], result4[i][2], result4[i][3], result4[i][4] = data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]
np.savetxt('.\\output\\stag.txt', result4)
#%% setup positions
# first cone
a, b, N1 = 0.146, 0.146 + 0.218, 2
interval = (b-a) / (N1-1)
x1 = np.linspace(a+interval/2, b-interval/2, N1-1)

# second cone
a, b, N2 = 0.364, 0.364 + 2.122, 6
interval = (b-a) / (N2-1)
x2 = np.linspace(a+interval/2, b-interval/2, N2-1)

# cylinder
a, b, N3 = 2.486 + 0.113 + 0.848, 2.486 + 0.234 + 0.113 + 0.848, 4
interval = (b-a) / (N3-1)
x3 = np.linspace(a+interval/2, b-interval/2, N3-1)
#%% stag ht
from htcorrelate import propertyRatioHT, stagnationHT
heat_data_stag = np.zeros(shape=(ndata, 3))
Rnose = 0.04
for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    Mf, Tf, pf = result4[i][0], result4[i][4], result4[i][3]
    pfree = data[i][4]
    
    with HiddenPrints():
        heat_data_stag[i] = stagnationHT(Mf, Tf, pf, pfree, Rnose)
time = data[0:ndata,0]

heat_data_stag = np.column_stack([time, heat_data_stag])
np.savetxt('.\\output\\heat_data_stag.csv', heat_data_stag, delimiter=';') 

#%% ht cone 1
heat_data_c1 = np.zeros(shape=(ndata, (N1-1)*4 +1))
## Get Taw and HTC for conic
for i in range(0, ndata):
    Mf, Tf, pf = result1[i][0], result1[i][4], result1[i][3]
    heat_data_c1[i] = propertyRatioHT(Mf, Tf, pf, x1)
time = data[0:ndata,0]

heat_data_c1 = np.column_stack([time, heat_data_c1])

np.savetxt('.\\output\\heat_data_c1.csv', heat_data_c1, delimiter=';')

#%% ht cone 2
heat_data_c2 = np.zeros(shape=(ndata, (N2-1)*4 +1))
## Get Taw and HTC for conic
for i in range(0, ndata):
    Mf, Tf, pf = result2[i][0], result2[i][4], result2[i][3]
    heat_data_c2[i] = propertyRatioHT(Mf, Tf, pf, x2)
time = data[0:ndata,0]

heat_data_c2 = np.column_stack([time, heat_data_c2])

np.savetxt('.\\output\\heat_data_c2.csv', heat_data_c2, delimiter=';')

#%% ht cylinder
heat_data_cylinder = np.zeros(shape=(ndata, (N3-1)*4 +1))
for i in range(0, ndata):
    Mf, Tf, pf = result3[i][0], result3[i][4], result3[i][3]
    #with HiddenPrints():
    heat_data_cylinder[i] = propertyRatioHT(Mf, Tf, pf, x3, wedge='yes')
time = data[0:ndata,0]

heat_data_cylinder = np.column_stack([time, heat_data_cylinder])

np.savetxt('.\\output\\heat_data_cylinder.csv', heat_data_cylinder, delimiter=';')


############## WING
#%%  Solve oblique
from htshock2 import oshock, oshock_NRS

alpha = 7.3868 / 2                      # wedge angle
result21 = np.zeros(shape=(ndata, 5))    # result for after oblique

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if data[i][1] > 1.16:
        with HiddenPrints():
            result21[i][0], result21[i][1], result21[i][2], result21[i][3], result21[i][4], _ = oshock_NRS(data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], alpha)
    else:   # subsonic region does not change significantly.
        print('OBLIQUE SUBSONIC')
        with HiddenPrints():
            result21[i][0], result21[i][1], result21[i][2], result21[i][3], result21[i][4] = data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]

np.savetxt('.\\output\\obliquewing.txt', result21)

#%%  Solve oblique
from htshock2 import pmexpand

result22 = np.zeros(shape=(ndata, 5))    # result for after pm

for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    if result21[i][0] > 1.15:
        with HiddenPrints():
            result22[i][0], result22[i][1], result22[i][2], result22[i][3], result22[i][4] = pmexpand(result21[i][0], result21[i][1], result21[i][2], result21[i][3], result21[i][4], alpha, errmax=0.5)
    else:   # subsonic region does not change significantly.
        print('PM SUBSONIC')
        with HiddenPrints():
            result22[i][0], result22[i][1], result22[i][2], result22[i][3], result22[i][4] = result21[i][0], result21[i][1], result21[i][2], result21[i][3], result21[i][4]

np.savetxt('.\\output\\pmwing.txt', result22)
#%% wedge
a, b, N1 = 38.67e-3, 38.67e-3 + 207.03e-3, 4
interval = (b-a) / (N1-1)
x1 = np.linspace(a+interval/2, b-interval/2, N1-1)

a, b, N2 = 38.67e-3 + 207.03e-3, 38.67e-3 + 207.03e-3 + 531.96e-3, 6
interval = (b-a) / (N2-1)
x2 = np.linspace(a+interval/2, b-interval/2, N2-1)

heat_data_wedge = np.zeros(shape=(ndata, (N1-1)*4 +1))
for i in range(0, ndata):
    Mf, Tf, pf = result21[i][0], result21[i][4], result21[i][3]
    heat_data_wedge[i] = propertyRatioHT(Mf, Tf, pf, x1, wedge='yes')
time = data[0:ndata,0]

heat_data_wedge = np.column_stack([time, heat_data_wedge])

#%% cylin
heat_data_wingcylinder = np.zeros(shape=(ndata, (N2-1)*4 +1))
for i in range(0, ndata):
    Mf, Tf, pf = result22[i][0], result22[i][4], result22[i][3]
    with HiddenPrints():
        heat_data_wingcylinder[i] = propertyRatioHT(Mf, Tf, pf, x2, wedge='yes')
time = data[0:ndata,0]

heat_data_wingcylinder = np.column_stack([time, heat_data_wingcylinder])

#%% stag
heat_data_wingstag = np.zeros(shape=(ndata, 3))
Rnose = 2.49E-3
for i in range(0, ndata):
    print('time: ', data[i][0], '/', data[-1][0])
    Mf, Tf, pf = result4[i][0], result4[i][4], result4[i][3]
    pfree = data[i][4]
    
    with HiddenPrints():
        heat_data_wingstag[i] = stagnationHT(Mf, Tf, pf, pfree, Rnose)
time = data[0:ndata,0]

heat_data_wingstag = np.column_stack([time, heat_data_wingstag])