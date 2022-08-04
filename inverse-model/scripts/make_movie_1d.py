# This program creates a group of png images of the data, inversion, and "true" solution
# over time. After running this, change to the 'pngs_2'
# directory and run the following script, where frames_per_sec is an integer (e.g., 50):
#
# ffmpeg -r frames_per_sec -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4
#
# (This requires ffmpeg: https://ffmpeg.org/)

import sys
sys.path.insert(0, '../source')
sys.path.insert(0, '../data')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from params import H


if os.path.isdir('../pngs')==False:
    os.mkdir('../pngs')    # make a directory for the results.

# Load relevant files
resultsname = 'results'

# "data" and true solution
wb = np.load('../data/wb.npy')             # Lower surface
h = np.load('../data/h.npy')               # Upper surface
dV = np.load('../data/dV.npy')             # Upper surface

# inversion and associated forward elevation, volume change estimate
w_inv = np.load('../results/w_inv.npy')             # Lower surface
h_fwd = np.load('../results/h_fwd.npy')               # Upper surface
dV_inv = np.load('../results/dV_inv.npy')             # Upper surface

# Create array for plotting
x = np.load('../data/x.npy')                          # x-coordinate array
t = np.load('../data/t.npy')

for i in range(np.size(t)):
    print('image '+str(i)+' out of '+str(np.size(t)))

    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.plot(t[0:i],dV[0:i]*H,color='royalblue',linewidth=3,label=r'true sol.')
    plt.plot(t[0:i],dV_inv[0:i]*H,color='k',linestyle='--',linewidth=3,label=r'inversion')

    plt.ylabel(r'$\Delta V$ (m$^2$)',fontsize=20)
    plt.xlim(0,t[-1])
    plt.ylim(-50*1000,50*1000)
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(4,4),useMathText=True)
    plt.gca().yaxis.offsetText.set_fontsize(16)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12,loc='lower right')


    plt.subplot(222)
    plt.plot(x,h[i,:],color='royalblue',linewidth=3,label=r'data')
    plt.plot(x,h_fwd[i,:],color='k',linestyle='--',linewidth=3,label=r'model')
    plt.ylabel(r'$h$ (m)',fontsize=20)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.tick_right()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_label_position("right")
    plt.ylim(-2,2)
    plt.legend(fontsize=12,loc='lower right')


    plt.subplot(224)
    plt.plot(x,wb[i,:],color='royalblue',linewidth=3,label=r'true sol.')
    plt.plot(x,w_inv[i,:],color='k',linestyle='--',linewidth=3,label=r'inversion')
    plt.ylabel(r'$w_b$ (m/yr)',fontsize=20)
    plt.ylim(-15,5)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12,loc='lower right')
    plt.tight_layout()
    plt.savefig('../pngs/'+str(i))
    plt.close()
