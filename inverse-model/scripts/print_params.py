# print the auxiliary model parameters

import sys
sys.path.insert(0, '../source')

from params import *

def print_params():
    print('auxiliary model parameters:')
    print('H = '+str(H/1e3)+' km')
    print('beta = '+'{:.2e}'.format(beta_d)+' Pa s/m')
    print('eta = '+'{:.2e}'.format(eta_d)+' Pa s')
    print('u = '+str(u_d)+' m/yr')
    print('v = '+str(v_d)+' m/yr')
