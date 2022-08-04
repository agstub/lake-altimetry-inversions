from scipy.integrate import trapezoid,cumulative_trapezoid
from params import dx,dt,x
import numpy as np

def calc_dV(w,L):
    # calculate water volume change estimate from inversion
    w[np.abs(x)>0.5*L]=0
    dV = cumulative_trapezoid(trapezoid(w,dx=dx,axis=-1),dx=dt,initial=0)
    return dV
