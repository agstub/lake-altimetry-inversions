from scipy.integrate import trapezoid,cumulative_trapezoid
from params import dx,dy,dt,x,y,ind,t
import numpy as np

def calc_dV_w(w,w_bdry):
    # calculate water volume change estimate from inversion
    w_copy = np.copy(w)*w_bdry
    dV = cumulative_trapezoid(trapezoid(trapezoid(w_copy,dx=dy,axis=-1),dx=dx,axis=-1),dx=dt,initial=0)
    return dV

def calc_dV_h(h,h_bdry):
    # calculate water volume change estimate from inversion
    ind_0 = np.argmax(ind[:,0,0])

    ind_r = np.copy(ind)
    ind_r[t>0.5*np.max(t)] = 1

    h_copy = np.copy(h)*h_bdry*ind_r
    h0 = (np.copy(h)[ind_0,:,:] + 0*np.copy(h))*h_bdry*ind_r
    dV = trapezoid(trapezoid(h_copy-h0,dx=dy,axis=-1),dx=dx,axis=-1)
    return dV
