#-------------------------------------------------------------------------------
# This function defines the rate of water volume change in the subglacial lake
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scm
from params import nt,t_period,t_final,Lngth
from geometry import interface,bed
from scipy.integrate import quad

d0 = 0.15            # Smoothing parameter

# Smoothed triangle wave
def trg(t):
    return 1 - 2*np.arccos((1 - d0)*np.sin(2*np.pi*t))/np.pi

# Smooth square wave
def sqr(t):
    return 2*np.arctan(np.sin(2*np.pi*t)/d0)/np.pi

# Smoothed sawtooth wave
def swt(t):
    return (1 + trg((2*t - 1)/4)*sqr(t/2))/2

lake_vol_0 = 2*np.pi*quad(lambda x: (interface(x)-bed(x))*x,0,0.5*Lngth,full_output=1)[0]

# Sawtooth volume change
def Vol(t):
    V = lake_vol_0*(1.5*swt(t/t_period)+0.5*swt(0))
    return V

def Vdot(t):
    # compute rate of subglacial lake volume change
    dt_fine = 3.154e7/5000.0       # timestep for computing derivative (1/5000 yr)
    Vd = scm.derivative(Vol,t,dx=dt_fine)
    return Vd



## ------------------------------
##plot lake volume timeseries:
# import matplotlib.pyplot as plt
# t = np.linspace(0,t_final,nt)
#
# plt.plot(t/3.154e7,(Vol(t)-Vol(0))/1e9)
# #plt.axhline(y=lake_vol_0/1e9,linestyle='--')
# plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
# plt.xlabel(r'$t$ (yr)',fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.show()
