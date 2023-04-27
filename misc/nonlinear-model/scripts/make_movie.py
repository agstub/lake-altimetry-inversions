# This program creates a group of png images of the elevation anomaly and basal
# surface-normal velocity over time. After running this, change to the 'pngs'
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
from geometry import interface, bed
from params import tol,t_final,Lngth,Hght,t_period,C,nt
from hydrology import Vol

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

if os.path.isdir('pngs')==False:
    os.mkdir('pngs')    # make a directory for the results.

# Load relevant files
resultsname = 'results'

wb = np.load('data_nonlinear/w_true.npy')           # Lower surface
h_obs = np.load('data_nonlinear/h_nloc.npy')        # Upper surface contour
dV = np.load('data_nonlinear/dV_true.npy')          # Volume change


# Create array for plotting
H = np.load('data_nonlinear/H.npy').mean()        # thickness
x = np.load('data_nonlinear/x_d.npy')             # x-coordinate array
y = np.load('data_nonlinear/y_d.npy')             # x-coordinate array
t = np.load('data_nonlinear/t.npy')

# Load relevant files
resultsname = 'results'

s = np.loadtxt(resultsname+'/s')             # Lower surface radial
h = np.loadtxt(resultsname+'/h')             # Upper surface radial


# Create array for plotting
X = np.loadtxt(resultsname+'/r')             # x-coordinate array
t_f = np.loadtxt(resultsname+'/t')           # t-coordinate array


dV_max = np.max(dV*(H**2)/1e9)+0.1
dV_min = np.min(dV*(H**2)/1e9)-0.1

for i in range(np.size(t)):

    j = np.argmin(np.abs(t_f/3.154e7-t[i]))
    print('image '+str(i)+' out of '+str(np.size(t)))

    plt.figure(figsize=(8,12))

#-------------------------------------------------------------------------------
    plt.subplot(311)
    plt.title(r'$t=$'+format(t[i],'.2f')+' yr',loc='left',fontsize=24)

    # Plot upper surface
    plt.plot(X/1000,h[:,j]-0.98*Hght,color='royalblue',linewidth=2,label=r'$h$')
    plt.plot(-X/1000,h[:,j]-0.98*Hght,color='royalblue',linewidth=2,label=r'$h$')


    # Plot ice, water, and bed domains; colored accordingly.
    p1 = plt.fill_between(X/1000,y1=s[:,j], y2=h[:,j]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
    p2 = plt.fill_between(X/1000,bed(X),s[:,j],facecolor='slateblue',alpha=0.5)
    p3 = plt.fill_between(X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)

    p1 = plt.fill_between(-X/1000,y1=s[:,j], y2=h[:,j]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
    p2 = plt.fill_between(-X/1000,bed(X),s[:,j],facecolor='slateblue',alpha=0.5)
    p3 = plt.fill_between(-X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)


    # Plot bed surface
    plt.plot(X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
    plt.plot(X[(s[:,j]-bed(X)>1e-3*tol)]/1000,s[:,j][(s[:,j]-bed(X)>1e-3*tol)],'o',color='crimson',markersize=1,label=r'$s>\beta$')

    plt.plot(-X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
    plt.plot(-X[(s[:,j]-bed(X)>1e-3*tol)]/1000,s[:,j][(s[:,j]-bed(X)>1e-3*tol)],'o',color='crimson',markersize=1,label=r'$s>\beta$')


    # Label axes and save png:
    plt.xlabel(r'$r$ (km)',fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=16)
    plt.xlim(-0.5*Lngth/1000,0.5*Lngth/1000)
    plt.ylim(np.min(bed(X))-2,36)


#-------------------------------------------------------------------------------

    plt.subplot(312)
    plt.plot(t[0:i],dV[0:i]*(H**2)/1e9,color='k',linewidth=3)
    plt.plot([t[i]],[dV[i]*(H**2)/1e9],color='k',marker='o',markersize=10)
    #plt.plot(t_f[0:j]/3.154e7,(Vol(t_f[0:j])-Vol(0))/1e9,color='k',linestyle='--',linewidth=3)
    plt.ylabel(r'$\Delta V$ (km$^3$)',fontsize=20)
    plt.xlim(0,t[-1])
    plt.ylim(dV_min,dV_max)
    plt.xlabel(r'$t$ (yr)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.subplot(325)
    p1 = plt.contourf(x,y,h_obs[i,:,:].T,cmap='PuOr_r',levels=np.linspace(-1,1,6),extend='both')
    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(p1,orientation='horizontal',pad=0.25)
    cbar.set_label(r'$\Delta h$ (m)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)



    plt.subplot(326)
    p2 = plt.contourf(x,y,wb[i,:,:].T,cmap='PuOr_r',extend='both',levels=np.linspace(-1,1,6))
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    # Label axes and save png:
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.gca().yaxis.set_ticklabels([])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(p2,orientation='horizontal',pad=0.25)
    cbar.set_label(r'$w_b$ (m/yr)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('pngs/'+str(i))
    plt.close()
