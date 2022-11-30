# This script plots some locations of subglacial lakes on top of the
# ICESat-2 ATL15 data map

import sys
sys.path.insert(0, '../source')

from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from load_lakes import gdf
from matplotlib.colors import to_hex

fn = '../../../ICESat-2/ATL15/ATL15_AA_0311_01km_001_01.nc'
ds = nc.Dataset(fn)
dsh = ds['delta_h']

dh = dsh['delta_h'][:]        # elevation change (m)
sigma = dsh['delta_h_sigma'][:]
x = dsh['x'][:]               # x coordinate array (m)
y = dsh['y'][:]               # y coordinate array (m)
mask = dsh['ice_mask'][:]
t = dsh['time'][:]               # y coordinate array (m)

# Byrd_s10 coordinates
lake_name = 'Byrd_s10'
outline = gdf.loc[gdf['name']==lake_name]
xB = float(outline.centroid.x)*1e3
yB = float(outline.centroid.y)*1e3


# Slessor_23
lake_name = 'Slessor_23'
outline = gdf.loc[gdf['name']==lake_name]
xS = float(outline.centroid.x)*1e3
yS = float(outline.centroid.y)*1e3

# Mac1
lake_name = 'Mac1'
outline = gdf.loc[gdf['name']==lake_name]
xMa = float(outline.centroid.x)*1e3
yMa = float(outline.centroid.y)*1e3


# Mercer
lake_name = 'MercerSubglacialLake'
outline = gdf.loc[gdf['name']==lake_name]
xMe = float(outline.centroid.x)*1e3
yMe = float(outline.centroid.y)*1e3

# Thwaites_170
lake_name = 'Thw_170'
outline = gdf.loc[gdf['name']==lake_name]
xT = float(outline.centroid.x)*1e3
yT = float(outline.centroid.y)*1e3



points_x = np.array([xB,xS,xT,xMa,xMe])/1e3
points_y = np.array([yB,yS,yT,yMa,yMe])/1e3

dh_0 = dh[-1,:,:]-dh[0,:,:]

plt.figure(figsize=(16,16))
plt.title(r'ICESat-2 ATL15 (07/2021 $-$ 10/2018)',fontsize=35)
p1 = plt.contourf(x/1000,y/1000,dh_0,cmap='coolwarm',levels=np.linspace(-1,1,11),extend='both')
plt.plot([points_x[-1]],[points_y[-1]],marker=r'*',color='k',markersize=40)
plt.plot([points_x[0]],[points_y[0]],marker=r'*',color='k',markersize=40)
plt.plot([points_x[1]],[points_y[1]],marker=r'*',color='k',markersize=40)
plt.plot([points_x[2]],[points_y[2]],marker=r'*',color='k',markersize=40)
plt.plot([points_x[3]],[points_y[3]],marker=r'*',color='k',markersize=40)


plt.plot([points_x[-1]],[points_y[-1]],marker=r'*',color='mediumseagreen',markersize=25)
plt.plot([points_x[0]],[points_y[0]],marker=r'*',color='mediumseagreen',markersize=25)
plt.plot([points_x[1]],[points_y[1]],marker=r'*',color='mediumseagreen',markersize=25)
plt.plot([points_x[2]],[points_y[2]],marker=r'*',color='mediumseagreen',markersize=25)
plt.plot([points_x[3]],[points_y[3]],marker=r'*',color='mediumseagreen',markersize=25)
plt.contour(x/1000,y/1000,mask,colors='k',linewidths=0.5)
plt.annotate(xy=(points_x[-1]-125,points_y[-1]-250),text=r'$\mathrm{\mathbf{SLM}}$',fontsize=22,color='k')
plt.annotate(xy=(points_x[0]+150,points_y[0]-50),text=r'$\rm{\mathbf{Byrd_{s10}}}$',fontsize=22,color='k')
plt.annotate(xy=(points_x[1]+150,points_y[1]-50),text=r'$\mathrm{\mathbf{Slessor_{23}}}$',fontsize=22,color='k')
plt.annotate(xy=(points_x[2]+150,points_y[2]-50),text=r'$\mathrm{\mathbf{Thw_{170}}}$',fontsize=22,color='k')
plt.annotate(xy=(points_x[3]-150,points_y[3]-225),text=r'$\mathrm{\mathbf{Mac1}}$',fontsize=22,color='k')

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel(r'$y$ (km)',fontsize=24)
plt.xlabel(r'$x$ (km)',fontsize=24)
plt.gca().set_aspect('equal', 'box')
cbar = plt.colorbar(p1,ticks=np.linspace(-1,1,11),orientation='vertical',fraction=0.035, pad=0.04)
cbar.set_label(r'$\Delta h$ (m)', fontsize=35)
cbar.ax.tick_params(labelsize=20)
plt.savefig('map',bbox_inches='tight')
plt.close()
