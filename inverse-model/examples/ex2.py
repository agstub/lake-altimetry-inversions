import sys
sys.path.insert(0, '../data')
sys.path.insert(0, '../source')


from inversion import invert
from operators import forward_w
from params import x,t,Nx,Nt
import numpy as np
from kernel_fcns import ifftd,fftd
from conj_grad import norm


L = 10
sigma = L/3


h = np.load('../data/h.npy')
w_true = np.load('../data/wb.npy')

# noise_h = np.random.normal(size=(Nt,Nx))
# noise_level = 0.0
# h += noise_level*norm(h)*noise_h/norm(noise_h)

import matplotlib.pyplot as plt
#plt.subplot(211)
p = plt.contourf(t,x,h,cmap='coolwarm')
plt.xlabel(r'$t$',fontsize=20)
plt.ylabel(r'$x$',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(p)
plt.tight_layout()
plt.show()
#plt.savefig('ex.png')

# sol,fwd,mis = invert(h,5e-3)
#
# w_min = np.min(w_true)
# w_max = np.max(w_true)
#
#
# import matplotlib.pyplot as plt
# plt.subplot(211)
# plt.contourf(sol.T,vmin=w_min,vmax=w_max,extend='both')
# plt.subplot(212)
# plt.contourf(w_true.T,vmin=w_min,vmax=w_max,extend='both')
# plt.show()
