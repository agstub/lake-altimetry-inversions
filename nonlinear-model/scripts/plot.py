# This file plots the free surface geometry at lake lowstand and highstand.
# The difference between these provides an estimate of the lake volume change,
# which we compare the true lake volume change to the volume change estimated from
# surface displacement.

import sys
sys.path.insert(0, './source')

import matplotlib.pyplot as plt
import numpy as np
from params import tol,t_final,nt_per_cycle,nt,t_period,Hght,Lngth,C
import subprocess
from scipy.misc import derivative
from scipy.interpolate import interp1d
import matplotlib as mpl

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

resultsname = 'results_t'+format(t_period/3.154e7,'.1f')+'_H'+format(Hght/1000.0,'.1f')+'_C'+str(int(np.floor(np.log10(C))))

# Load relevant files
Gamma_s = np.loadtxt(resultsname+'/Gamma_s')             # lower surface
Gamma_h = np.loadtxt(resultsname+'/Gamma_h')             # upper surface
x_left = np.loadtxt(resultsname+'/x_left')               # left grounding line
x_right = np.loadtxt(resultsname+'/x_right')             # right grounding line
X = np.loadtxt(resultsname+'/X')                         # x-coordinate array
V = np.loadtxt(resultsname+'/lake_vol')                  # lake volume change


ih = int(nt - nt_per_cycle - 1)                     # time index of highstand
il = int(nt - 0.5*nt_per_cycle - 1)                 # time index of lowstand

dH = Gamma_h[:,ih]-Gamma_h[:,il]                    # difference between highstand
                                                    # and lowstand elevations

Vest = np.trapz(dH[dH>=0.1],X[dH>=0.1])             # Volume change estime:
                                                    # 10 cm displacement threshold

Vtrue = V[ih]-V[il]                                 # true lake volume change

dV = Vest/Vtrue                                     # estimated vs. true volume
                                                    # change ratio

plt.figure(figsize=(8,6))
plt.subplot(211)
plt.title(r'$V_\mathrm{est}\,/\,V_\mathrm{true} = $'+format(dV,'.2f'),fontsize=24)
plt.plot(X/1000-0.5*Lngth/1000,Gamma_h[:,ih]-Hght,'-',color='royalblue',linewidth=2,label=r'highstand (hs)')
plt.plot(X/1000-0.5*Lngth/1000,Gamma_h[:,il]-Hght,'-',color='forestgreen',linewidth=2,label=r'lowstand (ls)')
plt.ylabel(r'$h-h_0$ (m)',fontsize=20)
plt.xticks(fontsize=16)
plt.gca().xaxis.set_ticklabels([])
plt.yticks(fontsize=16)
plt.legend(fontsize=16,loc='lower right')

plt.subplot(212)
plt.plot(X/1000-0.5*Lngth/1000,dH,'-',color='k',linewidth=2)
plt.fill_between(X[dH>0.1]/1000-0.5*Lngth/1000,y1=0*X[dH>0.1],y2=dH[dH>0.1],facecolor='crimson',alpha=0.5,label=r'$V_\mathrm{est}$')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(r'$h_{hs} - h_{ls}$ (m)',fontsize=20)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.legend(fontsize=16,loc='upper right')

plt.tight_layout()
plt.savefig('V_est')
plt.close()
