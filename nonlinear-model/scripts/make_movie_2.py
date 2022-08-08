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

wb = np.load('data_nonlinear/wb.npy')           # Lower surface
h = np.load('data_nonlinear/h.npy')             # Upper surface
dV = np.load('data_nonlinear/dV.npy')           # Upper surface


# Create array for plotting
H = np.load('data_nonlinear/H.npy').mean()      # thickness
x = np.load('data_nonlinear/x.npy')             # x-coordinate array
y = np.load('data_nonlinear/y.npy')             # x-coordinate array
t = np.load('data_nonlinear/t.npy')

# T,Y,X = np.meshgrid(t,y,x,indexing='ij')
#
#
# from scipy.integrate import trapezoid,cumulative_trapezoid
# def calc_dV_w(w,w_bdry):
#     # calculate water volume change estimate from inversion
#     w_copy = np.copy(w)*w_bdry
#     dV = cumulative_trapezoid(trapezoid(trapezoid(w_copy,dx=y[1]-y[0],axis=-1),dx=x[1]-x[0],axis=-1),dx=t[1]-t[0],initial=0)
#     return dV
#
# dV0 = calc_dV_w(wb,w_bdry)

for i in range(np.size(t)):
    print('image '+str(i)+' out of '+str(np.size(t)))

    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.plot(t[0:i],dV[0:i]*(H)/1e9,color='forestgreen',linewidth=3)
    plt.plot([t[i]],[dV[i]*(H)/1e9],color='forestgreen',marker='o',markersize=10)
    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t[-1])
    plt.ylim(-0.2,0.2)

    #plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.subplot(222)
    plt.contourf(x,y,h[i,:,:].T,cmap='coolwarm',levels=np.arange(-1,1,0.2),extend='both')
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.tick_right()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_label_position("right")


    plt.subplot(224)
    plt.contourf(x,y,wb[i,:,:].T,cmap='coolwarm',extend='both',levels=np.arange(-10,10,2))
    plt.ylabel(r'$y$ (km)',fontsize=20)

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
