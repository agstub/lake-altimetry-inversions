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
import os


if os.path.isdir('pngs_2')==False:
    os.mkdir('pngs_2')    # make a directory for the results.

# Load relevant files
resultsname = 'results'

wb = np.load('data/wb.npy')             # Lower surface
h = np.load('data/h.npy')             # Upper surface
dV = np.load('data/dV.npy')             # Upper surface


# Create array for plotting
x = np.load('data/x.npy')                          # x-coordinate array
t = np.load('data/t.npy')

for i in range(np.size(t)):
    print('image '+str(i)+' out of '+str(np.size(t)))

    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.plot(t[0:i],dV[0:i]*100,color='royalblue',linewidth=3)
    plt.plot([t[i]],[dV[i]*100],color='crimson',marker='o',markersize=15)
    plt.ylabel(r'$\Delta V$ (%)',fontsize=20)
    plt.xlim(0,t[-1])
    plt.ylim(-100,100)
    #plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.subplot(222)
    plt.plot(x,h[i,:],color='royalblue',linewidth=3)
    plt.ylabel(r'$h$ (m)',fontsize=20)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.tick_right()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_label_position("right")
    plt.ylim(-2,2)


    plt.subplot(224)
    plt.plot(x,wb[i,:],color='royalblue',linewidth=3)
    plt.ylabel(r'$w_b$ (m/yr)',fontsize=20)
    plt.ylim(-15,5)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    # Label axes and save png:
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('pngs_2/'+str(i))
    plt.close()
