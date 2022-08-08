# This program creates a group of png images of the upper and lower surfaces
# over time. After running this, change to the 'pngs_1' directory and run the
# following bash script, where frames_per_sec is an integer (e.g., 50):
#
# ffmpeg -r frames_per_sec -f image2 -s 1920x1080 -i %01d.png -vcodec libx264 -pix_fmt yuv420p -vf scale=1280:-2 movie.mp4
#
# (This requires ffmpeg: https://ffmpeg.org/)

import sys
sys.path.insert(0, './source')

import matplotlib.pyplot as plt
import numpy as np
from geometry import interface, bed
from params import tol,t_final,Lngth,Hght,t_period,C,nt
import matplotlib as mpl
import os

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

# Create directory for png's
if os.path.isdir('pngs_1')==False:
    os.mkdir('pngs_1')    # make a directory for the results.

# Load relevant files
resultsname = 'results'

s = np.loadtxt(resultsname+'/s')             # Lower surface
h = np.loadtxt(resultsname+'/h')             # Upper surface


# Create array for plotting
X = np.loadtxt(resultsname+'/r')                          # x-coordinate array
t = np.loadtxt(resultsname+'/t')

# Loop over time steps
for i in range(nt):

    print('image '+str(i)+' out of '+str(nt))

    plt.figure(figsize=(8,4))


    plt.title(r'$t=$'+format(t[i]/3.154e7,'.2f')+' yr',loc='left',fontsize=20)

    # Plot upper surface
    plt.plot(X/1000,h[:,i]-0.98*Hght,color='royalblue',linewidth=2,label=r'$h$')
    plt.plot(-X/1000,h[:,i]-0.98*Hght,color='royalblue',linewidth=2,label=r'$h$')


    # Plot ice, water, and bed domains; colored accordingly.
    p1 = plt.fill_between(X/1000,y1=s[:,i], y2=h[:,i]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
    p2 = plt.fill_between(X/1000,bed(X),s[:,i],facecolor='slateblue',alpha=0.5)
    p3 = plt.fill_between(X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)

    p1 = plt.fill_between(-X/1000,y1=s[:,i], y2=h[:,i]-0.98*Hght,facecolor='aliceblue',alpha=1.0)
    p2 = plt.fill_between(-X/1000,bed(X),s[:,i],facecolor='slateblue',alpha=0.5)
    p3 = plt.fill_between(-X/1000,-18*np.ones(np.size(X)),bed(X),facecolor='burlywood',alpha=1.0)


    # Plot bed surface
    plt.plot(X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
    plt.plot(X[(s[:,i]-bed(X)>1e-3*tol)]/1000,s[:,i][(s[:,i]-bed(X)>1e-3*tol)],'o',color='crimson',markersize=1,label=r'$s>\beta$')

    plt.plot(-X/1000,bed(X),color='k',linewidth=1,label=r'$\beta$')
    plt.plot(-X[(s[:,i]-bed(X)>1e-3*tol)]/1000,s[:,i][(s[:,i]-bed(X)>1e-3*tol)],'o',color='crimson',markersize=1,label=r'$s>\beta$')


    plt.annotate(r'air',xy=(-35,20.5),fontsize=16)
    plt.annotate(r'ice',xy=(-35,4.5),fontsize=16)
    plt.annotate(r'bed',xy=(-35,2),fontsize=16)
    plt.annotate(r'water',xy=(-4,-3.5),fontsize=16)


    # Label axes and save png:
    plt.xlabel(r'$r$ (km)',fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=16)
    plt.xlim(-0.5*Lngth/1000,0.5*Lngth/1000)
    plt.ylim(np.min(bed(X))-2,26)


    plt.savefig('pngs_1/'+str(i),bbox_inches='tight')
    plt.close()
