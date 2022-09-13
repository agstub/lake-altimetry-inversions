import sys
sys.path.insert(0, '../source')

from scipy.interpolate import griddata
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr

if os.path.isdir('../data')==False:
    os.mkdir('../data')

# Subglacial Lake Mercer coordinates
x0 = -292*1e3
y0 = -500*1e3

L0 = 20*1000
x_min = x0-L0
x_max = x0+L0
y_min = y0-L0
y_max = y0+L0

# import the data
fn = '../../../ICESat-2/ATL15/ATL15_AA_0311_01km_001_01.nc'
ds = nc.Dataset(fn)
dsh = ds['delta_h']

dh = dsh['delta_h'][:]        # elevation change (m)
sigma = dsh['delta_h_sigma'][:]
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
    f_far = f_far.sum(axis=(1,2))/(f_far != 0).sum(axis=(1,2))
    f_loc = f- np.multiply.outer(f_far,np.ones(np.shape(f[0,:,:])))
    return f_loc

dh_loc = localize(dh_f)
far = np.mean(dh_f-dh_loc,axis=(1,2))

#--------------------------PLOTTING-------------------------------

levels=np.arange(-6,6.1,0.5)

# plot png at each time step to make sure the interpolation worked

if os.path.isdir('data_pngs')==False:
    os.mkdir('data_pngs')

X_sub,Y_sub = np.meshgrid(x_sub,y_sub)

#-------------------------------------------------------------------------------
## PLOT elevation change anomaly
# for i in range(np.size(t_f)):
#     print('image '+str(i+1)+' out of '+str(np.size(t_f)))
#     plt.close()
#     plt.figure(figsize=(8,6))
#     plt.title(r'$t=$ '+'{:.2f}'.format(t_f[i])+' d',fontsize=24)
#     p = plt.contourf(x_f/1e3,y_f/1e3,dh_loc[i,:,:],levels=levels,cmap='coolwarm',extend='both')
#     plt.xlabel(r'$x$ (km)',fontsize=20)
#     plt.ylabel(r'$y$ (km)',fontsize=20)
#     cbar = plt.colorbar(p)
#     cbar.set_label(r'$dh$ (m)',fontsize=20)
#     cbar.ax.tick_params(labelsize=16)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.tight_layout()
#     plt.savefig('data_pngs/dh_'+str(i))
#     plt.close()

#---------------- thickness and drag estimates

H_beta = xr.open_zarr('../../../thickness/data/H_beta.zarr',consolidated=False)
H_beta.load()

x, y = np.array(H_beta.x),np.array(H_beta.y)                # horizontal map coordinates
beta = H_beta.beta.data                               # (dimensional) basal sliding coefficient (Pa s / m)
H = H_beta.thickness.data                           # ice thickness (m)

ind_x = np.arange(0,np.size(x),1)
ind_y = np.arange(0,np.size(y),1)

x_sub = x[(x>=x_min)&(x<=x_max)]
y_sub = y[(y>=y_min)&(y<=y_max)]

inds_x = ind_x[(x>=x_min)&(x<=x_max)]
inds_y = ind_y[(y>=y_min)&(y<=y_max)]

inds_xy = np.ix_(inds_x,inds_y)

X, Y = np.meshgrid(x,y)

H_mean = np.mean(H[inds_xy])
beta_mean = np.mean(beta[inds_xy])

np.save('../data/beta.npy',np.array([beta_mean]))
np.save('../data/H.npy',np.array([H_mean]))
np.save('../data/u.npy',np.array([0]))
np.save('../data/h_obs.npy',dh_loc)
np.save('../data/t.npy',(t_f-t_f[0])/365.0)
np.save('../data/x.npy',(x_f-x_f.mean())/H_mean)
np.save('../data/y.npy',(y_f-y_f.mean())/H_mean)
np.save('../data/eta.npy',np.array([1e13]))             # viscosity?!
