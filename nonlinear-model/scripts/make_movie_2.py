# This program creates a group of png images of the elevation anomaly and basal
# surface-normal velocity over time. After running this, change to the 'pngs_2'
# directory and run the following script, where frames_per_sec is an integer (e.g., 50):
#
# ffmpeg -r frames_per_sec -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4
#
# (This requires ffmpeg: https://ffmpeg.org/)

import sys
sys.path.insert(0, './source')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from params import nt,t0,dt,Lngth
import os


if os.path.isdir('pngs_2')==False:
    os.mkdir('pngs_2')    # make a directory for the results.

# Load relevant files
resultsname = 'results'

wb = np.loadtxt(resultsname+'/wb')             # Lower surface
h = np.loadtxt(resultsname+'/h')             # Upper surface


# Create array for plotting
x = np.loadtxt(resultsname+'/x')                          # x-coordinate array
t = np.loadtxt(resultsname+'/t')
lake_vol = np.loadtxt(resultsname+'/lake_vol')

i0 = int(t0/dt)

wb = wb-np.outer(wb[:,i0],np.ones(np.size(t)))
h = h-np.outer(h[:,i0],np.ones(np.size(t)))

wb = wb[:,i0:None]
h = h[:,i0:None]

V = (lake_vol[i0:None]-lake_vol[i0])/lake_vol[i0]

x0 = (x-0.5*Lngth)/1e3
t0 = t/3.154e7

for i in range(nt-i0):
    print('image '+str(i)+' out of '+str(nt-i0))

    plt.figure(figsize=(8,10))

    plt.subplot(311)
    plt.plot(t0[0:i],V[0:i],color='royalblue',linewidth=3)
    plt.ylabel(r'$\Delta V$',fontsize=16)
    plt.xlim(0,t0[-1])
    plt.ylim(-1,1)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.subplot(312)
    plt.plot(x0,h[:,i],color='royalblue',linewidth=3)
    plt.ylabel(r'$h$ (m)',fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.ylim(-2,2)


    plt.subplot(313)
    plt.plot(x0,wb[:,i],color='royalblue',linewidth=3)
    plt.ylabel(r'$w_b$ (m/yr)',fontsize=16)
    plt.ylim(-10,10)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    # Label axes and save png:
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('pngs_2/'+str(i))
    plt.close()
