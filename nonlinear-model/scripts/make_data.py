import sys
sys.path.insert(0, './source')
import numpy as np
from scipy.interpolate import interp2d,interp1d
from params import t_final,Lngth,Hght,nt
from geometry import bed
import matplotlib.pyplot as plt
import os

if os.path.isdir('./data_nonlinear')==False:
    os.mkdir('./data_nonlinear')    # make a directory for the ../results.

H = Hght - bed(0.5*Lngth)


t = np.loadtxt('./results/t')

lake_vol = np.loadtxt('./results/lake_vol')
dV = (lake_vol-lake_vol[0])/(H**2)

h = np.loadtxt('./results/h')

h = h - np.outer(h[:,0],np.ones(nt))

wb = np.loadtxt('./results/wb')            # saved in m/yr

r = np.loadtxt('./results/r')/H               # center and scale
t = t/3.154e7                             # trim and scale

eta = np.loadtxt('./results/eta_mean')
beta = np.loadtxt('./results/beta_mean')
u = np.loadtxt('./results/u_mean')              # saved in m/yr


h_int = interp2d(r, t, h.T)
wb_int = interp2d(r, t, wb.T)
dV_int = interp1d(t, dV)


nx_d = 101
nt_d = 200

x_d = np.linspace(-r[-1],r[-1],nx_d)
y_d = np.linspace(-r[-1],r[-1],nx_d)
t_d = np.linspace(t[0],t[-1],nt_d)

T_d,Y_d,X_d = np.meshgrid(t_d,y_d,x_d,indexing='ij')

h_d = 0*X_d
wb_d = 0*Y_d

for l in range(np.size(t_d)):
    for i in range(np.size(y_d)):
        for j in range(np.size(x_d)):
            h_d[l,i,j] = h_int(np.sqrt(X_d[l,i,j]**2 + Y_d[l,i,j]**2),T_d[l,i,j])
            wb_d[l,i,j] = wb_int(np.sqrt(X_d[l,i,j]**2 + Y_d[l,i,j]**2),T_d[l,i,j])


dV_d = dV_int(t_d)


# save numpy files for use in inversion
np.save('./data_nonlinear/dV.npy',dV_d)
np.save('./data_nonlinear/wb.npy',wb_d)
np.save('./data_nonlinear/h.npy',h_d)
np.save('./data_nonlinear/x.npy',x_d)
np.save('./data_nonlinear/y.npy',x_d)
np.save('./data_nonlinear/t.npy',t_d)
np.save('./data_nonlinear/H.npy',np.array([H]))
np.save('./data_nonlinear/beta.npy',beta)
np.save('./data_nonlinear/eta.npy',eta)


# # PLOT time series of mean viscosity and basal drag
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
