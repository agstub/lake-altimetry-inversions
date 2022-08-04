from scipy.integrate import trapezoid,cumulative_trapezoid
from params import dx,dt,x
import numpy as np

def calc_dV(w,L):
    # calculate water volume change estimate from inversion
    w_copy = np.copy(w)
    w_copy[np.abs(x)>L]=0
    dV = cumulative_trapezoid(trapezoid(w_copy,dx=dx,axis=-1),dx=dt,initial=0)
    return dV
