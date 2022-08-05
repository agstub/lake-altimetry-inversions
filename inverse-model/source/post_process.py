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
