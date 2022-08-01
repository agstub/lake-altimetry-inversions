#-------------------------------------------------------------------------------
# This function defines the rate of water volume change in the subglacial lake.
# *Default volume change = sinusoidal timeseries
# (of course you can calculate dVol/dt analytically for the default example)
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scm
from params import nt,t_period,t_final,t0

d0 = 0.1            # Smoothing parameter

# Smoothed triangle wave
def trg(t):
    return 1 - 2*np.arccos((1 - d0)*np.sin(2*np.pi*t))/np.pi


# Smooth square wave
def sqr(t):
    return 2*np.arctan(np.sin(2*np.pi*t)/d0)/np.pi

# Smoothed sawtooth wave
def swt(t):
    return (1 + trg((2*t - 1)/4)*sqr(t/2))/2

# Sawtooth volume change
def Vol(t,lake_vol_0):
    if t < t0:
        V = lake_vol_0+0*t
    else:
        V = 2*lake_vol_0*swt((t-t0)/t_period)
    return V

def Vdot(lake_vol_0,t):
    # compute rate of subglacial lake volume change
    dt_fine = 3.154e7/5000.0       # timestep for computing derivative (1/5000 yr)
    Vd = scm.derivative(Vol,t,dx=dt_fine,args=(lake_vol_0,))
    return Vd

#
# import matplotlib.pyplot as plt
# t = np.linspace(0,t_final,nt)
# V = 0*t
#
# for i in range(np.size(t)):
#     V[i] = Vol(t[i],1)
#
# plt.plot(V)
# plt.show()
