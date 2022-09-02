from scipy.integrate import trapezoid,cumulative_trapezoid
from params import dx,dy,dt,x,y
import numpy as np

def calc_dV_w(w,w_bdry):
    # calculate water volume change estimate from inversion
    w_copy = np.copy(w)*w_bdry
    dV = cumulative_trapezoid(trapezoid(trapezoid(w_copy,dx=dy,axis=-1),dx=dx,axis=-1),dx=dt,initial=0)
    return dV

def calc_dV_h(h,h_bdry):
    # calculate water volume change estimate from inversion
    h_copy = np.copy(h)*h_bdry
    dV = trapezoid(trapezoid(h_copy,dx=dy,axis=-1),dx=dx,axis=-1)
    return dV

def calc_bdry_w(w,thresh):
    bdry = 0*x
    bdry[np.abs(w)>thresh*np.max(np.abs(w))] = 1
    bdry[np.sqrt(x**2+y**2)>x.max()-0] = 0
    bdry = np.mean(bdry,axis=0)
    bdry[bdry<0.6] = 0
    bdry[bdry>=0.6] = 1
    return bdry

def calc_bdry_h(h,thresh):
    bdry = 1+0*x
    bdry[np.abs(h)<0.1] = 0
    bdry = np.mean(bdry,axis=0)
    bdry[bdry<0.25] = 0
    bdry[bdry>=0.25] = 1
    return bdry
