import sys
sys.path.insert(0, '../source')
import numpy as np
from scipy.interpolate import interp2d,interp1d
from params import t_final,i0,t0,Lngth,Hght,nt
from geometry import bed
import matplotlib.pyplot as plt
import os

if os.path.isdir('../data_nonlinear')==False:
    os.mkdir('../data_nonlinear')    # make a directory for the ../results.

H = Hght - bed(0)

t = np.loadtxt('../results/t')

lake_vol = np.loadtxt('../results/lake_vol')
dV = lake_vol[i0:None]-lake_vol[i0]

h = np.loadtxt('../results/h')
h = h-np.outer(h[:,i0],np.ones(np.size(t)))
h = h[:,i0:None]

wb = np.loadtxt('../results/wb')            # saved in m/yr
wb = wb-np.outer(wb[:,i0],np.ones(np.size(t)))
wb = wb[:,i0:None]

x = (np.loadtxt('../results/x')-Lngth/2.0)/H               # center and scale
t = (t[i0:None]-t0)/3.154e7                             # trim and scale

eta = np.loadtxt('../results/eta_mean')[i0:None]
beta = np.loadtxt('../results/beta_mean')[i0:None]
u = np.loadtxt('../results/u_mean')[i0:None]               # saved in m/yr


# # nondimensional parameters for inversion (reference vs mean value)
# print(beta[0]*H/(2*eta[0]))
# print(np.mean(beta)*H/(2*eta[0]))
#
# print(u[0]/H)
# print(np.mean(u)/H)


#
h_int = interp2d(x, t, h.T)
wb_int = interp2d(x, t, wb.T)
dV_int = interp1d(t,dV)

nx_d = 101
nt_d = 200


x_d = np.linspace(x[0],x[-1],nx_d)
t_d = np.linspace(t[0],t[-1],nt_d)

h_d = h_int(x_d,t_d)
wb_d = wb_int(x_d,t_d)
dV_d = dV_int(t_d)

# save numpy files for use in inversion
np.save('../data_nonlinear/wb.npy',wb_d)
np.save('../data_nonlinear/h.npy',h_d)
np.save('../data_nonlinear/x.npy',x_d)
np.save('../data_nonlinear/t.npy',t_d)
np.save('../data_nonlinear/dV.npy',dV_d)
np.save('../data_nonlinear/H.npy',np.array([H]))
np.save('../data_nonlinear/beta.npy',beta)
np.save('../data_nonlinear/eta.npy',eta)
np.save('../data_nonlinear/u.npy',u)






# # PLOT time series of mean viscosity, basal drag, and horizontal surface velocity
# #
# plt.figure(figsize=(6,10))
# plt.subplot(311)
# plt.plot(t,(eta-eta[0])/eta[0],color='royalblue',linestyle='none',marker='o',markersize=5)
# plt.ylabel(r'$\bar{\eta}\,/\,\bar{\eta_0} -1$',fontsize=20)
# plt.yticks(fontsize=16)
# plt.gca().xaxis.set_ticklabels([])
# plt.ylim(-0.15,0.15)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.subplot(312)
# plt.plot(t,(u-u[0])/u[0],color='royalblue',marker='o',linestyle='none',markersize=5)
# plt.ylabel(r'$\bar{u}\,/\,\bar{u}_0-1$',fontsize=20)
# plt.yticks(fontsize=16)
# plt.gca().xaxis.set_ticklabels([])
# plt.ylim(-0.15,0.15)
# # plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.subplot(313)
# plt.plot(t,(beta-beta[0])/beta[0],color='royalblue',marker='o',linestyle='none',markersize=5)
# plt.ylabel(r'$\bar{\beta}\,/\,\bar{\beta}_0 - 1$',fontsize=20)
# # plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
# # Label axes and save png:
# plt.xlabel(r'$t$ (yr)',fontsize=20)
# plt.ylim(-0.15,0.15)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.savefig('refs')
# plt.close()