from scipy.integrate import trapezoid,cumulative_trapezoid
from params import dx,dy,dt,x,y,t
import numpy as np

def calc_dV_w(w,w_bdry):
    # calculate water volume change estimate from inversion
    dV = cumulative_trapezoid(trapezoid(trapezoid(w*w_bdry,dx=dy,axis=-1),dx=dx,axis=-1),dx=dt,initial=0)
    return dV

def calc_dV_h(h,h_bdry):
    # calculate water volume change estimate from inversion
    dV = trapezoid(trapezoid(h*h_bdry,dx=dy,axis=-1),dx=dx,axis=-1)
    dV -= dV[0]
    return dV
