#-------------------------------------------------------------------------------
# This script preprocesses the ICESat-2 data for use in the inversion
#
# OVERVIEW
# The main choices are specifying:
# (I) lake_name (str): name of lake from the inventory:
#
#   Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial lake
#   activity in Antarctica from multi-mission satellite altimetry. Annals of
#   Glaciology, 59(76pt1), 42-55.
#
# (II) L0 (float):  half-length(/half-width) of horizontal domain (a box) surrounding
#                   the subglacial lake that was selected in the first step
#
# *There is also a function below ("localize") that removes the off-lake component
#  of the elevation change at each timestep
#
#
# DATA REQUIREMENTS:
# The following data are needed to run the script as-is:
# 1. ICESat-2 ATL15 (1 km resolution):
#    *website: https://nsidc.org/data/atl15/versions/2
#    *filename: ATL15_AA_0314_01km_002_02.nc
#
# 2. WAVI model output (for basal drag, ice thickness, viscosity )
#    *website: https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=5f0ac285-cca3-4a0e-bcbc-d921734395ab
#    *filename: WAVI_5km_Initial.nc
#    *reference:
#       Arthern, R. J., Hindmarsh, R. C., & Williams, C. R. (2015). Flow speed within
#       the Antarctic ice sheet and its controls inferred from satellite observations.
#       Journal of Geophysical Research: Earth Surface, 120(7), 1171-1188.
#
# 3. MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 (NSIDC-0754)
#    *webiste: https://nsidc.org/data/nsidc-0754/versions/1
#    *filename: antarctica_ice_vel_phase_map_v01.nc
#    *reference:
#        Mouginot, J., E. Rignot, and B. Scheuchl. (2019). MEaSUREs Phase-Based Antarctica 
#        Ice Velocity Map, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow 
#        and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PZ3NJ5RXRH10.
#
# 4. Antarctic Subglacial Lake inventory
#    *website: https://github.com/mrsiegfried/Siegfried2021-GRL
#    *filename: SiegfriedFricker2018-outlines.h5
#    *references:
#       Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial
#       lake activity in Antarctica from multi-mission satellite altimetry.
#       Annals of Glaciology, 59(76pt1), 42-55.
#
#       Siegfried, M. R., & Fricker, H. A. (2021). Illuminating active
#       subglacial lake processes with ICESat‐2 laser altimetry. Geophysical
#       Research Letters, 48(14), e2020GL091089.
#-------------------------------------------------------------------------------
import sys
sys.path.insert(0, '../source')

from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from load_lakes import gdf

# STEP (I): Select lake from Siegfried and Fricker (2018) inventory
lake_name = 'Mac1'
outline = gdf.loc[gdf['name']==lake_name]
data_name = 'data_'+lake_name
if os.path.isdir('../'+data_name)==False:
    os.mkdir('../'+data_name)
x0 = float(outline.centroid.x)*1e3
y0 = float(outline.centroid.y)*1e3

# STEP (II): Select half-width L0 of box surrounding lake
L0 = 30*1000
x_min = x0-L0
x_max = x0+L0
y_min = y0-L0
y_max = y0+L0

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# load the ATL15 data
fn = '../../../ICESat-2/ATL15/ATL15_AA_0314_01km_002_02.nc'
ds = nc.Dataset(fn)
dsh = ds['delta_h']

dh = dsh['delta_h'][:]        # elevation change (m)
x = dsh['x'][:]               # x coordinate array (m)
y = dsh['y'][:]               # y coordinate array (m)
t = dsh['time'][:]            # t coordinate array (d)

nt = np.size(t)

ind_x = np.arange(0,np.size(x),1)
ind_y = np.arange(0,np.size(y),1)

# extract the data that is inside the bounding box
x_sub = x[(x>=x_min)&(x<=x_max)]
y_sub = y[(y>=y_min)&(y<=y_max)]
inds_x = ind_x[(x>=x_min)&(x<=x_max)]
inds_y = ind_y[(y>=y_min)&(y<=y_max)]

nx = np.size(inds_x)
ny = np.size(inds_y)

inds_xy = np.ix_(inds_y,inds_x)
dh_sub = np.zeros((nt,ny,nx))

# put elevation change maps into 3D array with time being the first index
for i in range(nt):
    dh0 = dh[i,:,:]
    dh_sub[i,:,:] = dh0[inds_xy]

#--------------------------PLOTTING-------------------------------

levels=np.arange(-1,1.1,0.1)*np.max(np.abs(dh_sub))

# plot png at each time step

if os.path.isdir('../'+data_name+'/data_pngs')==False:
    os.mkdir('../'+data_name+'/data_pngs')

X_sub,Y_sub = np.meshgrid(x_sub,y_sub)


# PLOT elevation change anomaly
for i in range(np.size(t)):
    print('image '+str(i+1)+' out of '+str(np.size(t)))
    plt.close()
    plt.figure(figsize=(8,6))
    plt.title(r'$t=$ '+'{:.2f}'.format(t[i])+' d',fontsize=24)
    p = plt.contourf(X_sub/1e3,Y_sub/1e3,dh_sub[i,:,:],levels=levels,cmap='coolwarm',extend='both')
    outline.plot(edgecolor='k',facecolor='none',ax=plt.gca(),linewidth=3)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.ylabel(r'$y$ (km)',fontsize=20)
    cbar = plt.colorbar(p)
    cbar.set_label(r'$dh$ (m)',fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('../'+data_name+'/data_pngs/dh_'+str(i))
    plt.close()

##------------------------------------------------------------------------------
## INTERPOLATE DATA

def interp_tyx(f,t,y,x):
    Nx_f = 101            # fine Nx
    Ny_f = 101            # fine Ny
    Nt_f = 100            # fine Nt

    t0_f = np.linspace(t.min(),t.max(),num=Nt_f)  # fine time array
    x0_f = np.linspace(x.min(),x.max(),num=Nx_f)  # fine x coordinate array
    y0_f = np.linspace(y.min(),y.max(),num=Ny_f)  # fine y coordinate array
    t_f,y_f,x_f = np.meshgrid(t0_f,y0_f,x0_f,indexing='ij')

    points = (t_f,y_f,x_f)

    f_fine = griddata((t.ravel(),y.ravel(),x.ravel()),f.ravel(),points)

    return f_fine,t0_f,y0_f,x0_f



t_g,y_g,x_g = np.meshgrid(t,y_sub,x_sub,indexing='ij')

dh_f,t_f,y_f,x_f = interp_tyx(dh_sub,t_g,y_g,x_g)

t,y,x = np.meshgrid(t_f,y_f,x_f,indexing='ij')

def localize(f):
    f_far = np.copy(f)
    f_far[np.sqrt((x-x.mean())**2+(y-y.mean())**2)<0.8*np.sqrt((x-x.mean())**2+(y-y.mean())**2).max()] = 0
    F = (f_far != 0).sum(axis=(1,2))+1e-10
    f_far = f_far.sum(axis=(1,2))/F
    f_loc = f- np.multiply.outer(f_far,np.ones(np.shape(f[0,:,:])))
    return f_loc

dh_loc = localize(dh_f)


#---------------- thickness and drag estimates ---------------------------------

fn = '../../../WAVI/WAVI_5km_Initial.nc'
ds = nc.Dataset(fn)
x = ds['x'][:]                                   # m
y = ds['y'][:]                                   # m
H = ds['thickness'][:]                           # ice thickness (m)
beta = ds['Beta2'][:]*3.154e7                    # convert to Pa s/m
eta = ds['depth_averaged_viscosity'][:]*3.154e7  # convert to Pa s

ind_x = np.arange(0,np.size(x),1)
ind_y = np.arange(0,np.size(y),1)

x_sub = x[(x>=x_min)&(x<=x_max)]
y_sub = y[(y>=y_min)&(y<=y_max)]

inds_x = ind_x[(x>=x_min)&(x<=x_max)]
inds_y = ind_y[(y>=y_min)&(y<=y_max)]

inds_xy = np.ix_(inds_y,inds_x)

H_mean = np.mean(H[inds_xy])
beta_mean = np.mean(beta[inds_xy])
eta_mean = np.mean(eta[inds_xy])

#------------------------ surface velocity -------------------------------------
fn = '../../../measures/antarctica_ice_vel_phase_map_v01.nc'
ds = nc.Dataset(fn)
x = ds['x'][:]                  # m
y = ds['y'][:]                  # m
u = ds['VX'][:]                 # m/yr
v = ds['VY'][:]                 # m/yr


ind_x = np.arange(0,np.size(x),1)
ind_y = np.arange(0,np.size(y),1)

x_sub = x[(x>=x_min)&(x<=x_max)]
y_sub = y[(y>=y_min)&(y<=y_max)]

inds_x = ind_x[(x>=x_min)&(x<=x_max)]
inds_y = ind_y[(y>=y_min)&(y<=y_max)]

inds_xy = np.ix_(inds_y,inds_x)

u0 = u[inds_xy]
v0 = v[inds_xy]

u_mean = np.mean(u0)
v_mean = np.mean(v0)

# ----------------------------- SAVE DATA --------------------------------------
np.save('../'+data_name+'/eta.npy',np.array([eta_mean]))    # viscosity: Pa s
np.save('../'+data_name+'/beta.npy',np.array([beta_mean]))  # basal drag: Pa s / m
np.save('../'+data_name+'/H.npy',np.array([H_mean]))        # thickness: m
np.save('../'+data_name+'/h_obs.npy',dh_loc)                # elevation anomaly: m
np.save('../'+data_name+'/t.npy',(t_f-t_f[0])/365.0)        # time: yr
np.save('../'+data_name+'/u.npy',np.array([u_mean]))        # vel x: m/yr
np.save('../'+data_name+'/v.npy',np.array([v_mean]))        # vel y: m/yr

np.save('../'+data_name+'/x.npy',(x_f-x_f.mean())/H_mean)   # x coord (scaled)
np.save('../'+data_name+'/y.npy',(y_f-y_f.mean())/H_mean)   # y coord (scaled)

np.save('../'+data_name+'/x_d.npy',x_f/1e3)     # x coord. for plotting results
np.save('../'+data_name+'/y_d.npy',y_f/1e3)     # y coord. for plotting results
