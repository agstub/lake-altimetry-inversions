# this file contains a conjugate gradient method implementation that is used to solve
# the normal equations that arise from the least-squares minimization problem

from params import dx,dy,dt,cg_tol,max_cg_iter,t
from scipy.integrate import trapz
import numpy as np


# ------------ define inner products and norms for CG method--------------------
def prod(a,b):
    # inner product for the optimization problem: L^2(0,T;L^2) space-time inner product
    int = a*b
    p = trapz( trapz(trapz(int,dx=dx,axis=-2),dx=dy,axis=-1) ,dx=dt,axis=0)
    return p


def norm(a):
    # norm for the optimization problem
    return np.sqrt(prod(a,a))

#------------------------------------------------------------------------------

def cg_solve(A,b,tol = cg_tol):
# conjugate gradient method for solving the normal equations
#
#              A(X)  = b,           where...
#
# * A is a symmetric positive definite operator
# * b = right-side vector

    r0 = b                     # initial residual
    p = r0                     # initial search direction
    j = 1                      # iteration
    r = r0                     # initialize residual
    X = 0*p                    # initial guess

    rnorm0 = prod(r,r)    # (squared) norm of the residual: previous iteration
    rnorm1 = rnorm0       # (squared) norm of the residual: current iteration

    r00 = norm(b)
    Ap = A(p)
    d = prod(p,Ap)

    r0 = 0*r


    while np.sqrt(rnorm1)/r00 > tol:
        if j%10 == 0:
            print("CG iter. "+str(j)+': rel. residual norm = '+"{:.2e}".format(np.sqrt(rnorm1)/r00)+',  tol = '+"{:.2e}".format(tol)+' \r',end='')

        rnorm0 = prod(r,r)

        Ap = A(p)
        d = prod(p,Ap)

        alpha_c = rnorm0/d

        X = X + alpha_c*p                      # update solution

        r = r - alpha_c*Ap                    # update residual
        rnorm1 = prod(r,r)
        beta_c = rnorm1/rnorm0

        D0 = np.abs(prod(r0,r))/prod(r,r)

        if D0 > 0.5 and j>10:
            beta_c = 0.0
            r = b-A(X)

        p = r  + beta_c*p                      # update search direction
        j += 1

        r0 = r

        if j > max_cg_iter:
            print('\n...CG terminated because maximum iterations reached.')
            break
    if j<max_cg_iter:
        print('\n...CG converged!\n')

    return X
